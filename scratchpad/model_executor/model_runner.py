import gc
import json
from typing import Optional
import torch
from scratchpad.config import DeviceConfig, LoadConfig
from scratchpad.config.vllm_model_config import ModelConfig as VllmModelConfig
from scratchpad.distributed import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce,
)
from scratchpad.distributed.parallel_state import in_the_same_node_as
from .model_loader import get_model

from scratchpad.config.model_config import AttentionArch, ModelConfig
from scratchpad.nn.attention import FlashInferAttnBackend, TritonAttnBackend
from scratchpad.nn.layers.logits_processor import LogitsProcessorOutput
from scratchpad.nn.layers.sampler import Sampler
from scratchpad.scheduler.schedule_batch import global_args
from scratchpad.memory.pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)
from scratchpad.memory import (
    HeterogeneousMHATokenToKVPool,
    ReqToTokenPool,
    TokenToKVPoolAllocator,
    init_parameter_offload_manager,
)
from scratchpad.model_executor.forward_info import ForwardBatch
from scratchpad.model_executor.speculative.spec_info import SpeculativeAlgorithm
from scratchpad.server.args import ServerArgs
from scratchpad.utils import (
    get_available_gpu_memory,
    enable_show_time_cost,
    logger,
)
from scratchpad.constrained import disable_cache
from scratchpad.managers import ToppingsManager


class ModelRunner:
    """ModelRunner runs the forward passes of the models."""

    def __init__(
        self,
        model_config: ModelConfig,
        mem_fraction_static: float,
        gpu_id: int,
        tp_rank: int,
        tp_size: int,
        nccl_port: int,
        server_args: ServerArgs,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[TokenToKVPoolAllocator] = None,
    ):
        # Parse args
        self.model_config = model_config
        self.mem_fraction_static = mem_fraction_static
        self.device = server_args.device
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.dist_port = nccl_port
        self.server_args = server_args
        self.is_generation = model_config.is_generation
        self.is_multimodal = model_config.is_multimodal
        self.spec_algorithm = SpeculativeAlgorithm.NONE
        self.page_size = server_args.page_size
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        logger.info(f"model config: {model_config}")
        # Model-specific adjustment
        if (
            self.model_config.attention_arch == AttentionArch.MLA
            and not self.server_args.disable_mla
        ):
            logger.info("MLA optimization is turned on. Use triton backend.")
            self.server_args.attention_backend = "triton"

        if self.is_multimodal:
            logger.warning(
                "Automatically turn off --chunked-prefill-size and adjust --mem-fraction-static for multimodal models."
            )
            server_args.chunked_prefill_size = None
            self.mem_fraction_static *= 0.95
            # TODO: qwen2-vl does not support radix cache now, set disable_radix_cache=True automatically
            if self.model_config.hf_config.architectures == [
                "Qwen2VLForConditionalGeneration"
            ]:
                server_args.disable_radix_cache = True

        # Global vars
        if server_args.show_time_cost:
            enable_show_time_cost()
        if server_args.disable_disk_cache:
            disable_cache()

        global_args.update(
            {
                "attention_backend": server_args.attention_backend,
                "sampling_backend": server_args.sampling_backend,
                "triton_attention_reduce_in_fp32": server_args.triton_attention_reduce_in_fp32,
                "disable_mla": server_args.disable_mla,
                "torchao_config": server_args.torchao_config,
                "disable_penalizer": server_args.disable_penalizer,
                "disable_nan_detection": server_args.disable_nan_detection,
            }
        )

        # Init componnets

        min_per_gpu_memory = self.init_torch_distributed()
        self.sampler = Sampler()
        self.load_model()
        self.init_toppings_manager()
        self.init_parameter_offloading()
        self.init_memory_pool(
            min_per_gpu_memory,
            server_args.max_running_requests,
            server_args.max_total_tokens,
        )
        if self.device == "cuda":
            self.init_cublas()
            self.init_attention_backend()
            self.init_cuda_graphs()
        else:
            self.cuda_graph_runner = None
            self.init_attention_backend()

    def init_torch_distributed(self):
        logger.info("Init torch distributed begin.")
        # Init torch distributed
        if self.device == "cuda":
            torch.cuda.set_device(self.gpu_id)
            backend = "nccl"
        else:
            raise ValueError(f"Unsupported device: {self.device}")
        # if not self.server_args.enable_p2p_check:
        #     monkey_patch_vllm_p2p_access_check(self.gpu_id)
        if self.server_args.dist_init_addr:
            dist_init_method = f"tcp://{self.server_args.dist_init_addr}"
        else:
            dist_init_method = f"tcp://127.0.0.1:{self.dist_port}"

        set_custom_all_reduce(not self.server_args.disable_custom_all_reduce)

        init_distributed_environment(
            backend=backend,
            world_size=self.tp_size,
            rank=self.tp_rank,
            local_rank=self.gpu_id,
            distributed_init_method=dist_init_method,
        )
        initialize_model_parallel(tensor_model_parallel_size=self.tp_size)

        min_per_gpu_memory = get_available_gpu_memory(
            self.device, self.gpu_id, distributed=self.tp_size > 1
        )
        self.tp_group = get_tp_group()

        # Currently, there is a bug with mulit-node tensor parallelsim + padded cuda graph, so we disable padding in cuda graph.
        if self.device == "cuda" and not all(
            in_the_same_node_as(self.tp_group.cpu_group, source_rank=0)
        ):
            self.server_args.disable_cuda_graph_padding = True
            logger.info(
                "Setting disable_cuda_graph_padding to True because of multi-node tensor parallelism."
            )

        # Check memory for tensor parallelism
        if self.tp_size > 1:
            local_gpu_memory = get_available_gpu_memory(self.device, self.gpu_id)
            if min_per_gpu_memory < local_gpu_memory * 0.9:
                raise ValueError(
                    "The memory capacity is unbalanced. Some GPUs may be occupied by other processes."
                )

        return min_per_gpu_memory

    def load_model(self):
        logger.info(
            f"Load weight begin. avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        # This can reduce thread conflicts and speed up weight loading.
        torch.set_num_threads(1)
        if self.device == "cuda":
            if torch.cuda.get_device_capability()[0] < 8:
                logger.info(
                    "Compute capability below sm80. Use float16 due to lack of bfloat16 support."
                )
                self.server_args.dtype = "float16"
                if torch.cuda.get_device_capability()[1] < 5:
                    raise RuntimeError("SP only supports sm75 and above.")

        # Prepare the vllm model config
        self.load_config = LoadConfig(load_format=self.server_args.load_format)
        self.vllm_model_config = VllmModelConfig(
            model=self.server_args.model_path,
            quantization=self.server_args.quantization,
            tokenizer=None,
            tokenizer_mode=None,
            trust_remote_code=self.server_args.trust_remote_code,
            dtype=self.server_args.dtype,
            seed=self.server_args.random_seed,
            skip_tokenizer_init=True,
        )
        if self.model_config.model_override_args is not None:
            self.vllm_model_config.hf_config.update(
                self.model_config.model_override_args
            )
        self.dtype = self.vllm_model_config.dtype

        # Load the model
        self.model = get_model(
            model_config=self.vllm_model_config,
            load_config=self.load_config,
            device_config=DeviceConfig(self.device),
        )
        self.sliding_window_size = (
            self.model.get_attention_sliding_window_size()
            if hasattr(self.model, "get_attention_sliding_window_size")
            else None
        )

        logger.info(
            f"Load weight end. "
            f"type={type(self.model).__name__}, "
            f"dtype={self.dtype}, "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

    def update_weights(self, model_path: str, load_format: str):
        """Update weights in-place."""
        from .model_loader import (
            DefaultModelLoader,
            device_loading_context,
            get_model_loader,
        )
        from .utils import set_default_torch_dtype

        logger.info(
            f"Update weights begin. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        target_device = torch.device(self.device)

        try:
            # TODO: Use a better method to check this
            vllm_model_config = VllmModelConfig(
                model=model_path,
                quantization=self.server_args.quantization,
                tokenizer=None,
                tokenizer_mode=None,
                trust_remote_code=self.server_args.trust_remote_code,
                dtype=self.server_args.dtype,
                seed=self.server_args.random_seed,
                skip_tokenizer_init=True,
            )
        except Exception as e:
            message = f"Failed to load model config: {e}."
            return False, message

        load_config = LoadConfig(load_format=load_format)

        # Only support vllm DefaultModelLoader for now
        loader = get_model_loader(load_config)
        if not isinstance(loader, DefaultModelLoader):
            message = f"Failed to get model loader: {loader}."
            return False, message

        def get_weight_iter(config):
            iter = loader._get_weights_iterator(
                DefaultModelLoader.Source(
                    config.model,
                    revision=config.revision,
                    fall_back_to_pt=getattr(
                        self.model, "fall_back_to_pt_during_load", True
                    ),
                )
            )
            return iter

        def model_load_weights(model, iter):
            model.load_weights(iter)
            for _, module in self.model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    with device_loading_context(module, target_device):
                        quant_method.process_weights_after_loading(module)
            return model

        with set_default_torch_dtype(vllm_model_config.dtype):
            try:
                iter = get_weight_iter(vllm_model_config)
            except Exception as e:
                message = f"Failed to get weights iterator: {e}."
                return False, message
            try:
                model = model_load_weights(self.model, iter)
            except Exception as e:
                message = (
                    f"Failed to update weights: {e}.\nRolling back to original weights."
                )
                del iter
                gc.collect()
                iter = get_weight_iter(self.vllm_model_config)
                self.model = model_load_weights(self.model, iter)
                return False, message

        self.model = model
        self.server_args.model_path = model_path
        self.server_args.load_format = load_format
        self.vllm_model_config = vllm_model_config
        self.load_config = load_config
        self.model_config.path = model_path

        logger.info("Update weights end.")
        return True, "Succeeded to update model weights."

    def init_toppings_manager(self):
        self.topping_manager = ToppingsManager(
            self.server_args,
            base_model=self.model,
            base_hf_config=self.model_config.hf_config,
            load_config=self.load_config,
        )

    def profile_max_num_token(self, total_gpu_memory: int):
        available_gpu_memory = get_available_gpu_memory(
            self.device, self.gpu_id, distributed=self.tp_size > 1
        )
        if (
            self.model_config.attention_arch == AttentionArch.MLA
            and not self.server_args.disable_mla
        ):
            cell_size = (
                (self.model_config.kv_lora_rank + self.model_config.qk_rope_head_dim)
                * self.model_config.num_hidden_layers
                * torch._utils._element_size(self.kv_cache_dtype)
            )
        else:
            cell_size = (
                self.model_config.get_num_kv_heads(self.tp_size)
                * self.model_config.head_dim
                * self.model_config.num_hidden_layers
                * 2
                * torch._utils._element_size(self.kv_cache_dtype)
            )
        rest_memory = available_gpu_memory - total_gpu_memory * (
            1 - self.mem_fraction_static
        )
        max_num_token = int(rest_memory * (1 << 30) // cell_size)
        return max_num_token

    def init_memory_pool(
        self,
        total_gpu_memory: int,
        max_num_reqs: Optional[int] = None,
        max_total_tokens: Optional[int] = None,
    ):
        if self.server_args.kv_cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        elif self.server_args.kv_cache_dtype == "fp8_e5m2":
            self.kv_cache_dtype = torch.float8_e5m2
        else:
            raise ValueError(
                f"Unsupported kv_cache_dtype: {self.server_args.kv_cache_dtype}."
            )

        self.max_total_num_tokens = self.profile_max_num_token(total_gpu_memory)
        if max_total_tokens is not None:
            if max_total_tokens > self.max_total_num_tokens:
                logger.warning(
                    f"max_total_tokens={max_total_tokens} is larger than the profiled value "
                    f"{self.max_total_num_tokens}. "
                    f"Use the profiled value instead."
                )
            self.max_total_num_tokens = min(self.max_total_num_tokens, max_total_tokens)

        if self.max_total_num_tokens <= 0:
            raise RuntimeError(
                "Not enough memory. Please try to increase --mem-fraction-static."
            )

        if max_num_reqs is None:
            max_num_reqs = min(
                max(
                    int(
                        self.max_total_num_tokens / self.model_config.context_len * 512
                    ),
                    2048,
                ),
                4096,
            )

        self.req_to_token_pool = ReqToTokenPool(
            size=max_num_reqs + 1,
            max_context_len=self.model_config.context_len + 4,
            device=self.device,
            use_records=False,
        )
        if (
            self.model_config.attention_arch == AttentionArch.MLA
            and not self.server_args.disable_mla
        ):
            self.token_to_kv_pool = MLATokenToKVPool(
                self.max_total_num_tokens,
                dtype=self.kv_cache_dtype,
                kv_lora_rank=self.model_config.kv_lora_rank,
                qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                layer_num=self.model_config.num_hidden_layers,
                device=self.device,
            )
        else:
            self.token_to_kv_pool = MHATokenToKVPool(
                self.max_total_num_tokens,
                page_size=self.page_size,
                dtype=self.kv_cache_dtype,
                head_num=self.model_config.get_num_kv_heads(self.tp_size),
                head_dim=self.model_config.head_dim,
                layer_num=self.model_config.num_hidden_layers,
                device=self.device,
                enable_memory_saver=self.server_args.enable_memory_saver,
            )
        if self.token_to_kv_pool_allocator is None:
            if self.page_size > 1:
                raise NotImplementedError(f"page_size > 1 is not supported.")
            self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
                self.max_total_num_tokens,
                dtype=self.kv_cache_dtype,
                device=self.device,
                kvcache=self.token_to_kv_pool,
            )
        logger.info(
            f"Memory pool end. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

    def init_cublas(self):
        """We need to run a small matmul to init cublas. Otherwise, it will raise some errors later."""
        dtype = torch.float16
        device = "cuda"
        a = torch.ones((16, 16), dtype=dtype, device=device)
        b = torch.ones((16, 16), dtype=dtype, device=device)
        c = a @ b
        return c

    def init_attention_backend(self):
        """Init attention kernel backend."""
        if self.server_args.attention_backend == "flashinfer":
            self.attn_backend = FlashInferAttnBackend(self)
        elif self.server_args.attention_backend == "triton":
            assert self.sliding_window_size is None, (
                "Window attention is not supported in the triton attention backend. "
                "Please use `--attention-backend flashinfer`."
            )
            assert not self.model_config.is_encoder_decoder, (
                "Cross attention is not supported in the triton attention backend. "
                "Please use `--attention-backend flashinfer`."
            )
            self.attn_backend = TritonAttnBackend(self)
        else:
            raise ValueError(
                f"Invalid attention backend: {self.server_args.attention_backend}"
            )

    def init_double_sparsity_channel_config(self, selected_channel):

        selected_channel = "." + selected_channel + "_proj"
        self.sorted_channels = []
        # load channel config
        with open(self.server_args.ds_channel_config_path, "r") as f:
            channel_config = json.load(f)

        for i in range(self.model_config.num_hidden_layers):
            key = "model.layers." + str(i) + ".self_attn" + selected_channel
            self.sorted_channels.append(
                torch.tensor(channel_config[key])[
                    :, : self.server_args.ds_heavy_channel_num
                ]
                .contiguous()
                .cuda()
            )

    def init_cuda_graphs(self):
        """Capture cuda graphs."""
        from scratchpad.model_executor.cuda_graph_runner import CudaGraphRunner

        self.cuda_graph_runner = None

        if not self.is_generation:
            # TODO: Currently, cuda graph only captures decode steps, which only exists for generation models
            return

        if self.server_args.disable_cuda_graph:
            return

        logger.info("Capture cuda graph begin. This can take up to several minutes.")
        self.cuda_graph_runner = CudaGraphRunner(self)

    def forward_decode(self, forward_batch: ForwardBatch):
        if self.cuda_graph_runner and self.cuda_graph_runner.can_run(forward_batch):
            return self.cuda_graph_runner.replay(forward_batch)

        forward_batch.positions = (forward_batch.seq_lens - 1).to(torch.int64)
        self.attn_backend.init_forward_metadata(forward_batch)
        return self.model.forward(
            forward_batch.input_ids, forward_batch.positions, forward_batch
        )

    def forward_extend(self, forward_batch: ForwardBatch):
        self.attn_backend.init_forward_metadata(forward_batch)
        if self.is_generation:
            return self.model.forward(
                forward_batch.input_ids, forward_batch.positions, forward_batch
            )
        else:
            # Only embedding models have get_embedding parameter
            return self.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                get_embedding=True,
            )

    def forward(self, forward_batch: ForwardBatch) -> LogitsProcessorOutput:
        if forward_batch.forward_mode.is_decode():
            return self.forward_decode(forward_batch)
        elif forward_batch.forward_mode.is_extend():
            return self.forward_extend(forward_batch)
        else:
            raise ValueError(f"Invaid forward mode: {forward_batch.forward_mode}")

    def sample(
        self, logits_output: LogitsProcessorOutput, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        # Apply logit bias
        sampling_info = forward_batch.sampling_info
        if sampling_info.sampling_info_done:
            # Overlap mode: the function update_regex_vocab_mask was executed
            # in process_batch_result of the last batch.
            if sampling_info.grammars:
                sampling_info.sampling_info_done.wait()
        else:
            # Normal mode: Put CPU-heavy tasks here. They will be overlapped with the forward pass.
            sampling_info.update_regex_vocab_mask()
            sampling_info.update_penalties()
        sampling_info.apply_logits_bias(logits_output.next_token_logits)

        # Sample the next tokens
        next_token_ids = self.sampler(
            logits_output,
            forward_batch.sampling_info,
            forward_batch.return_logprob,
            forward_batch.top_logprobs_nums,
            forward_batch.token_ids_logprobs,
        )
        return next_token_ids

    @property
    def model_is_mrope(self) -> bool:
        """Detect if the model has "mrope" rope_scaling type.
        mrope requires keep "rope_deltas" between prompt and decoding phases."""
        rope_scaling = getattr(self.model_config.hf_config, "rope_scaling", {})
        if rope_scaling is None:
            return False
        return rope_scaling.get("type", None) == "mrope"

    def init_parameter_offloading(self):
        """Initialize parameter offloading if enabled in server args."""
        if not self.server_args.enable_cpu_offload:
            logger.info("Parameter CPU offloading is disabled")
            return

        # Initialize the parameter offload manager
        offload_ratio = self.server_args.cpu_offload_ratio
        prefetch_window = self.server_args.prefetch_window
        enable_prefetch = self.server_args.enable_prefetch

        # Create offload manager
        self.param_offload_manager = init_parameter_offload_manager(
            enable_offload=True,
            enable_prefetch=enable_prefetch,
            cpu_offload_ratio=offload_ratio,
            prefetch_window=prefetch_window,
            strict_device_match=self.server_args.strict_device_match,
        )

        # Process specific layer module names if provided
        offload_modules = []
        if self.server_args.offload_layer_modules:
            offload_modules = self.server_args.offload_layer_modules.split(",")

        # If specific modules are provided, register only those
        if offload_modules:
            for module_name in offload_modules:
                for name, module in self.model.named_modules():
                    if module_name in name:
                        # Extract layer number if possible for priority
                        layer_num = (
                            0  # Default value in case we can't find a layer number
                        )
                        parts = name.split(".")
                        for i, part in enumerate(parts):
                            if part.isdigit():
                                layer_num = int(part)
                                break

                        priority = (
                            layer_num + 1
                        )  # Higher layer number = higher priority
                        self.param_offload_manager.register_module(
                            name, module, priority=priority
                        )
        else:
            # Register transformer layers by default
            self.param_offload_manager.register_model_layers(self.model)

        # Start the offloading process
        self.param_offload_manager.start_offloading()

        # Register forward hooks to dynamically load parameters when needed
        self._register_offload_hooks()

        logger.info(
            f"Parameter offloading initialized with {offload_ratio*100:.1f}% parameters offloaded to CPU. "
            f"Prefetching is {'enabled' if enable_prefetch else 'disabled'}"
        )

    def _register_offload_hooks(self):
        """Register pre/post forward hooks for parameter offloading."""
        if (
            not hasattr(self, "param_offload_manager")
            or self.param_offload_manager is None
        ):
            return

        # Register pre-forward hooks for all offloaded modules
        for name, module in self.model.named_modules():
            if name in self.param_offload_manager.modules:
                # Create hook that captures module name
                def make_pre_hook(module_name):
                    def hook(module, input):
                        self.param_offload_manager.pre_forward_hook(module_name)

                    return hook

                def make_post_hook(module_name):
                    def hook(module, input, output):
                        self.param_offload_manager.post_forward_hook(module_name)

                    return hook

                # Register the hooks
                module.register_forward_pre_hook(make_pre_hook(name))
                module.register_forward_hook(make_post_hook(name))
