from transformers import AutoConfig
import internal.configs.llama as llama_config
from humanize import intword, naturalsize
from typing import TYPE_CHECKING, List
import math

if TYPE_CHECKING:
    from .request import GenerationRequest


class MemoryPlanner:
    def __init__(
        self,
        model_params: AutoConfig,
        hardware_params: dict,
        w_bit: int = 16,
        a_bit: int = 16,
        kv_bit: int = 16,
        gpu_utilization: float = 0.9,
        parallel_config=None,
        block_size: int = 16,
    ):
        """
        Initialize memory planner for KV cache management.

        Args:
            model_params: HuggingFace model configuration object
            hardware_params: Hardware parameter dictionary with memory and compute specs
            w_bit: Weight precision in bits
            a_bit: Activation precision in bits
            kv_bit: KV cache precision in bits
            gpu_utilization: Target GPU utilization (0.0-1.0)
            parallel_config: Tensor parallel configuration (not implemented)
            block_size: Number of tokens per memory block
        """
        self.model_params = model_params
        self.parallel_config = parallel_config
        self.hardware_params = hardware_params
        self.gpu_utilization = gpu_utilization
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.kv_bit = kv_bit
        self.block_size = block_size
        self._allocated_blocks = 0
        self._allocation_map = {}
        self._max_num_blocks = self.get_max_num_blocks()

    def get_max_num_blocks(self):
        """
        Calculate the maximum number of KV cache blocks that can be allocated.

        Returns:
            int: Maximum number of blocks available for KV cache allocation
        """
        # TODO(xiaozhe): we ignored the memory for activations
        # TODO(xiaozhe): add support for parallel configs
        total_memory = self.hardware_params["vmemory"]
        # memory for weights
        # per layer memory
        w_memory = self.get_weights_memory()

        # Check if weights exceed total memory
        if w_memory >= total_memory:
            print(
                f"Warning: Model weights ({naturalsize(w_memory)}) exceed GPU memory ({naturalsize(total_memory)}). No KV cache blocks available."
            )
            return 0

        block_memory_size = (
            2
            * self.block_size
            * llama_config.get_num_key_value_heads(self.model_params)
            * llama_config.get_head_dim(self.model_params)
            * self.kv_bit
            / 8
        )
        total_block_memory_size = (
            block_memory_size * llama_config.get_num_hidden_layers(self.model_params)
        )

        available_memory = total_memory - w_memory
        max_blocks = math.floor(available_memory / total_block_memory_size)

        # Ensure we don't return negative values
        return max(0, max_blocks)

    def get_weights_memory(self):
        """
        Calculate the memory required to store model weights.

        Returns:
            float: Total memory in bytes required for all model weights
        """
        mlp_weights = (
            3
            * llama_config.get_hidden_size(self.model_params)
            * llama_config.get_intermediate_size(self.model_params)
            * self.w_bit
            / 8
        )

        q_weights = (
            llama_config.get_hidden_size(self.model_params)
            * llama_config.get_num_attention_heads(self.model_params)
            * llama_config.get_head_dim(self.model_params)
        )

        kv_weights = (
            2
            * llama_config.get_hidden_size(self.model_params)
            * llama_config.get_head_dim(self.model_params)
            * llama_config.get_num_key_value_heads(self.model_params)
        )

        o_weights = (
            llama_config.get_hidden_size(self.model_params)
            * llama_config.get_num_attention_heads(self.model_params)
            * llama_config.get_head_dim(self.model_params)
        )
        self_attn_weights = (q_weights + kv_weights + o_weights) * self.w_bit / 8

        lm_head_weights = (
            llama_config.get_hidden_size(self.model_params)
            * llama_config.get_vocab_size(self.model_params)
            * self.w_bit
            / 8
        )
        embedding_weights = (
            llama_config.get_hidden_size(self.model_params)
            * llama_config.get_vocab_size(self.model_params)
            * self.w_bit
            / 8
        )

        return (
            (mlp_weights + self_attn_weights)
            * llama_config.get_num_hidden_layers(self.model_params)
            + lm_head_weights
            + embedding_weights
        )

    def print_status(self):
        """
        Print current memory allocation status for debugging.
        """
        print(
            f"Weights memory / Total memory: {naturalsize(self.get_weights_memory())} / {naturalsize(self.hardware_params['vmemory'])}"
        )
        print(
            f"Allocated blocks/Total blocks: {self._allocated_blocks}/{self._max_num_blocks}"
        )

    def can_allocate_request(self, request: "GenerationRequest"):
        """
        Check if there is enough memory to allocate blocks for a request.

        Args:
            request: The GenerationRequest to check allocation for

        Returns:
            bool: True if allocation is possible, False otherwise
        """
        # If no blocks are available, return False immediately
        if self._max_num_blocks == 0:
            return False

        if request.req_id not in self._allocation_map:
            # this is a new request
            num_required_blocks = math.ceil(request.input_length / self.block_size)
            return (
                self._max_num_blocks * 0.95 - self._allocated_blocks
                >= num_required_blocks
            )
        else:
            # at least one block is available
            return self._max_num_blocks - self._allocated_blocks >= 1

    def allocate(self, request: "GenerationRequest"):
        """
        Allocate memory blocks for a request's KV cache.

        Args:
            request: The GenerationRequest to allocate memory for
        """

        def _allocate_blocks(request_id: str, num_blocks: int):
            self._allocated_blocks += num_blocks
            if request_id not in self._allocation_map:
                self._allocation_map[request_id] = num_blocks
            else:
                self._allocation_map[request_id] += num_blocks
            assert (
                self._allocated_blocks <= self._max_num_blocks
            ), "Exceeding memory limit"

        if request.req_id not in self._allocation_map:
            num_required_blocks = math.ceil(request.input_length / self.block_size)
            _allocate_blocks(request.req_id, num_required_blocks)
            return
        num_tokens_reserved = self._allocation_map[request.req_id] * self.block_size
        num_tokens_required = max(0, request.generated_tokens - num_tokens_reserved)
        assert (
            num_tokens_required == 0 or num_tokens_required == 1
        ), f"num_tokens_required: {num_tokens_required}"

        if num_tokens_required == 0:
            return
        _allocate_blocks(request.req_id, 1)

    def free(self, request_ids: List[str]):
        """
        Free memory blocks allocated to completed requests.

        Args:
            request_ids: List of request IDs whose memory should be freed
        """
        for req_id in request_ids:
            num_blocks = self._allocation_map.pop(req_id)
            self._allocated_blocks -= num_blocks
        assert self._allocated_blocks >= 0, "Negative allocated blocks"
