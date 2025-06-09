import json
from dataclasses import dataclass
from typing import Optional, List, Union, Dict
from scratchpad.utils.logger import logger
import tempfile


@dataclass
class ServerArgs:
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    server_id: str = "default"
    device: str = "cuda"
    chat_template: Optional[str] = None
    model_path: str = ""
    api_key: Optional[str] = None
    served_model_name: str = "auto"
    trust_remote_code: bool = True
    # json_model_override_args: str = "{}"
    model_override_args: str = "{}"
    is_embedding: bool = False
    context_length: int = 4096
    skip_tokenizer_init: bool = False
    tokenizer_path: str = "auto"
    tokenizer_mode: str = "auto"
    schedule_policy: str = "lpm"
    random_seed: Optional[int] = None
    stream_interval: int = 1
    watchdog_timeout: float = 20
    decode_log_interval: int = 10
    # memory and scheduling
    chunked_prefill_size: int = 8192
    max_prefill_tokens: int = 16384
    max_running_requests: Optional[int] = None
    max_total_tokens: Optional[int] = None
    kv_cache_dtype: str = "auto"
    schedule_conservativeness: float = 1.0
    # model
    load_format: str = "auto"
    quantization: Optional[str] = None
    dtype: str = "auto"
    page_size: int = 1
    # parallelism
    dist_init_addr: Optional[str] = None
    dp_size: int = 1
    tp_size: int = 1
    nnodes: int = 1
    node_rank: int = 0
    load_balance_method: str = "round_robin"
    enable_memory_saver: bool = False
    # internal ports and and names
    scheduler_input_ipc_name: str = "auto"
    tokenizer_ipc_name: str = "auto"
    detokenizer_ipc_name: str = "auto"
    tokenizer_port: int = 30001
    scheduler_port: int = 30002
    detokenizer_port: int = 30003
    # note(xiaozhe): this is actually a list of ints, but can be provided as a comma-separated string
    nccl_ports: str = "30004"
    init_new_token_ratio: float = 0.7
    base_min_new_token_ratio: float = 0.1
    new_token_ratio_decay: float = 0.001
    num_continue_decode_steps: int = 10
    retract_decode_steps: int = 20
    mem_fraction_static: float = 0.8

    enable_dp_attention: bool = False
    allow_auto_truncate: bool = False
    # constrained
    constrained_json_whitespace_pattern: Optional[str] = None
    enable_custom_logit_processor: bool = False
    # tokenization
    skip_special_tokens_in_output = True
    spaces_between_special_tokens_in_out = True
    enable_precache_with_tracing = True
    enable_parallel_encoding = True

    # logging stats
    enable_stats_logging: bool = True

    # backend
    attention_backend: Optional[str] = None
    sampling_backend: Optional[str] = None
    grammar_backend: str = "xgrammar"  # or outlines
    cuda_graph_bs: Optional[List[int]] = None
    cuda_graph_max_bs: Optional[int] = None
    speculative_algorithm: Optional[str] = None
    enable_cache_report: bool = True
    # debugging

    enable_nan_detection: bool = True
    disable_flashinfer: bool = False
    disable_flashinfer_sampling: bool = False
    disable_radix_cache: bool = False
    disable_jump_forward: bool = False
    disable_cuda_graph: bool = False
    disable_cuda_graph_padding: bool = False
    disable_disk_cache: bool = False
    disable_custom_all_reduce: bool = False
    disable_mla: bool = False
    enable_mixed_chunk: bool = False
    enable_torch_compile: bool = False
    enable_dp_attention: bool = False
    max_torch_compile_bs: int = 32
    torchao_config: str = ""
    enable_p2p_check: bool = False
    flashinfer_workspace_size: int = 384 * 1024 * 1024
    triton_attention_reduce_in_fp32: bool = False
    triton_attention_num_kv_splits: int = 8
    log_requests: bool = False
    show_time_cost: bool = False
    disable_penalizer: bool = False
    num_continuous_decode_steps: int = 1
    # experimental
    enable_system_controller: bool = False
    use_heterogeneous_pool: bool = False
    controller_port: int = 30005
    enable_overlap_schedule: bool = True
    enable_double_sparsity: bool = False
    disable_nan_detection: bool = False
    # Topping config
    enable_toppings: bool = False
    max_toppings_per_batch: int = 4
    init_number_of_deltas: int = 1
    init_number_of_loras: int = 1
    max_lora_ranks: int = 64
    # comma separated list of toppings, format: type:identifier:served_name
    init_toppings: Optional[str] = None
    allow_toppings_registration: bool = False

    # CPU offloading settings
    enable_cpu_offload: bool = False
    cpu_offload_ratio: float = 0.7  # Percentage of parameters to offload
    enable_prefetch: bool = True  # Whether to prefetch parameters from CPU to GPU
    prefetch_window: int = 2  # Number of layers to prefetch ahead
    offload_layer_modules: str = "layers"
    strict_device_match: bool = (
        True  # Whether to enforce parameters are on expected device
    )

    # others
    crash_on_warnings: bool = False
    # internal use, no need to set (should be auto-generated)
    tool_call_parser: Optional[str] = "auto"
    reasoning_parser: Optional[str] = "auto"

    def translate_auto(self):
        if self.served_model_name == "auto":
            self.served_model_name = self.model_path
        if self.tokenizer_path == "auto":
            self.tokenizer_path = self.model_path
        if type(self.nccl_ports) == str:
            self.nccl_ports = self.nccl_ports.split(",")
        if self.attention_backend is None:
            self.attention_backend = "flashinfer"
        if self.sampling_backend is None:
            self.sampling_backend = "flashinfer"
        if self.random_seed is None:
            self.random_seed = 0
        if self.scheduler_input_ipc_name == "auto":
            self.scheduler_input_ipc_name = (
                f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
            )
        if self.tokenizer_ipc_name == "auto":
            self.tokenizer_ipc_name = (
                f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
            )
        if self.detokenizer_ipc_name == "auto":
            self.detokenizer_ipc_name = (
                f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
            )
        try:
            self.json_model_override_args = json.loads(self.model_override_args)
        except Exception as e:
            import traceback

            traceback.print_exc()
            logger.warning(f"Failed to parse model_override_args: {e}")
            self.json_model_override_args = {}

        if self.grammar_backend not in ["xgrammar", "outlines"]:
            raise ValueError(
                f"Invalid grammar_backend: {self.grammar_backend}, Expect one of ['xgrammar', 'outlines']"
            )
        if self.cuda_graph_bs is None:
            self.cuda_graph_max_bs = 160

        if self.tool_call_parser == "auto":
            if "llama" in self.served_model_name.lower():
                self.tool_call_parser = "llama3"
            elif "qwen3" in self.served_model_name.lower():
                self.tool_call_parser = "qwen3"
            else:
                self.tool_call_parser = None
            logger.info(f"Using tool_call_parser: {self.tool_call_parser}")

        if self.reasoning_parser == "auto":
            if "qwen3" in self.served_model_name.lower():
                self.reasoning_parser = "qwen3"
            else:
                self.reasoning_parser = None
            logger.info(f"Using reasoning_parser: {self.reasoning_parser}")

    def update(self, args):
        for k, v in args.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self.translate_auto()

    def check_experimental(self):
        if self.use_heterogeneous_pool:
            logger.warning(
                "--use_heterogeneous_pool is an experimental feature. Use with caution."
            )


global_args = ServerArgs()
