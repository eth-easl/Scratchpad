from dataclasses import dataclass, field
from typing import Optional, List, Union
from scratchpad.utils import Singleton


@dataclass
class ServerArgs(metaclass=Singleton):
    host: str = "0.0.0.0"
    port: int = 3000
    debug: bool = False

    model_path: str = ""
    served_model_name: str = "auto"
    trust_remote_code: bool = True
    json_model_override_args: str = "{}"
    is_embedding: bool = False
    context_length: int = 4096
    skip_tokenizer_init: bool = False
    tokenizer_path: str = "auto"
    tokenizer_mode: str = "auto"
    schedule_policy: str = "lpm"
    random_seed: Optional[int] = None
    # model
    load_format: str = "auto"
    quantization: Optional[str] = None
    dtype: str = "auto"
    # parallelism
    nccl_init_addr: Optional[str] = None
    dp_size: int = 1
    tp_size: int = 1
    nnodes: int = 1
    load_balance_method: str = "round_robin"
    # internal ports
    tokenizer_port: int = 30001
    controller_port: int = 30002
    detokenizer_port: int = 30003
    ## note(xiaozhe): this is actually a list of ints, but can be provided as a comma-separated string
    nccl_ports: str = "30004"
    #
    init_new_token_ratio: float = 0.7
    base_min_new_token_ratio: float = 0.1
    new_token_ratio_decay: float = 0.001
    num_continue_decode_steps: int = 10
    retract_decode_steps: int = 20
    mem_fraction_static: float = 0.9
    # tokenization
    skip_special_tokens_in_output = True
    spaces_between_special_tokens_in_out = True
    enable_precache_with_tracing = True
    enable_parallel_encoding = True

    # toppings config
    lora_paths: str = ""
    max_loras_per_batch: int = 1

    # debugging
    attention_backend: Optional[str] = None
    sampling_backend: Optional[str] = None
    disable_flashinfer: bool = False
    disable_flashinfer_sampling: bool = False
    disable_radix_cache: bool = False
    disable_regex_jump_forward: bool = False
    disable_cuda_graph: bool = False
    disable_cuda_graph_padding: bool = False
    disable_disk_cache: bool = False
    disable_custom_all_reduce: bool = False
    disable_mla: bool = False
    enable_mixed_chunk: bool = False
    enable_torch_compile: bool = False
    max_torch_compile_bs: int = 32
    torchao_config: str = ""
    enable_p2p_check: bool = False

    triton_attention_reduce_in_fp32: bool = False

    def translate_auto(self):
        if self.served_model_name == "auto":
            self.served_model_name = self.model_path
        if self.tokenizer_path == "auto":
            self.tokenizer_path = self.model_path
        if type(self.nccl_ports) == str:
            self.nccl_ports = self.nccl_ports.split(",")

    def update(self, args):
        for k, v in args.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self.translate_auto()


global_args = ServerArgs()
