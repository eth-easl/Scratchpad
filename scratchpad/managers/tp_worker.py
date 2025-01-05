import json

from attr import s
from scratchpad.utils import (
    broadcast_pyobj,
    is_multimodal_model,
    set_random_seed,
    logger,
    get_processor,
    get_tokenizer,
)
from scratchpad.server.args import ServerArgs
from scratchpad.model_executor.model_runner import ModelRunner
from scratchpad.config.model_config import ModelConfig
from scratchpad.scheduler.schedule_batch import ModelWorkerBatch
from scratchpad.memory.het_pool import HeterogeneousMHATokenToKVPool
from scratchpad.model_executor.forward_info import ForwardBatch
from .structs import UpdateWeightReqInput
from typing import Optional
from scratchpad.server.args import global_args


class TpModelWorker:
    """A tensor parallel model worker."""

    def __init__(
        self,
        gpu_id: int,
        tp_rank: int,
        server_args: ServerArgs,
        nccl_port: int,
        dp_rank: Optional[int] = 0,
    ):
        # Parse args
        logger.info(f"Initalizing model worker on GPU {gpu_id}, tp_rank: {tp_rank}")
        self.tp_rank = tp_rank
        self.server_args = server_args
        # Init model and tokenizer
        self.model_config = ModelConfig(
            server_args.model_path,
            server_args.trust_remote_code,
            context_length=server_args.context_length,
            model_override_args=server_args.json_model_override_args,
        )
        self.model_runner = ModelRunner(
            model_config=self.model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            tp_size=server_args.tp_size,
            nccl_port=nccl_port,
            server_args=server_args,
        )
        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if is_multimodal_model(self.model_config.hf_config.architectures):
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                )
                self.tokenizer = self.processor.tokenizer
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                )
        self.device = self.model_runner.device
        # Profile number of tokens
        self.max_total_num_tokens = self.model_runner.max_total_num_tokens
        self.max_prefill_tokens = server_args.max_prefill_tokens
        self.max_running_requests = min(
            (
                self.max_total_num_tokens // 2
                if server_args.max_running_requests is None
                else server_args.max_running_requests
            ),
            self.model_runner.req_to_token_pool.size,
        )
        self.max_req_len = min(
            self.model_config.context_len - 1,
            self.max_total_num_tokens - 1,
        )
        self.max_req_input_len = self.max_req_len - 5
        assert (
            self.max_req_len > 0 and self.max_req_input_len > 0
        ), "Memory pool size is too small"

        # Sync random seed across TP workers
        self.random_seed = broadcast_pyobj(
            [server_args.random_seed],
            self.tp_rank,
            self.model_runner.tp_group.cpu_group,
        )[0]
        set_random_seed(self.random_seed)

    def get_worker_info(self):
        return (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_req_len,
            self.max_req_input_len,
            self.random_seed,
            self.device,
            global_args,
            self.model_runner.req_to_token_pool.size,
            self.model_runner.req_to_token_pool.max_context_len,
            self.model_runner.token_to_kv_pool.size,
        )

    # (note:xiaozhe): deprecated
    def get_token_and_memory_info(self):
        return (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_req_input_len,
            self.random_seed,
        )

    def expand_memory_pool(self, increments: int):
        """
        Expand memory pool by `increments`.
        This is a no-op if the memory pool is not a `HeterogeneousMHATokenToKVPool`
        or `server_args.use_heterogeneous_pool is False`
        """
        if not self.server_args.use_heterogeneous_pool:
            logger.warning(
                "Not using heterogeneous pool, not possible to expand memory pool"
            )
            return
        if not isinstance(
            self.model_runner.token_to_kv_pool, HeterogeneousMHATokenToKVPool
        ):
            logger.warning(
                f"Memory pool is not a HeterogeneousMHATokenToKVPool, not possible to expand memory pool, got {type(self.model_runner.token_to_kv_pool)}"
            )
            return
        self.model_runner.expand_kv_pool(increments)

    def get_pad_input_ids_func(self):
        return getattr(self.model_runner.model, "pad_input_ids", None)

    def get_tp_cpu_group(self):
        return self.model_runner.tp_group.cpu_group

    def get_memory_pool(self):
        return (
            self.model_runner.req_to_token_pool,
            self.model_runner.token_to_kv_pool,
        )

    def forward_batch_generation(self, model_worker_batch: ModelWorkerBatch):
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        logits_output = self.model_runner.forward(forward_batch)
        next_token_ids = self.model_runner.sample(logits_output, model_worker_batch)
        return logits_output, next_token_ids

    def forward_batch_embedding(self, model_worker_batch: ModelWorkerBatch):
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        logits_output = self.model_runner.forward(forward_batch)
        embeddings = logits_output.embeddings
        return embeddings

    def update_weights(self, recv_req: UpdateWeightReqInput):
        success, message = self.model_runner.update_weights(
            recv_req.model_path, recv_req.load_format
        )
        return success, message

    def register_topping(self, topping):
        self.model_runner.topping_manager.register_topping(topping)
