import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Union

from humanize import intword, naturalsize
import internal.configs.llama as llama_config
from transformers import AutoConfig


@dataclass(frozen=True)
class ParallelConfig:
    """Describe model parallel settings for an engine."""

    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1

    def with_defaults(self) -> "ParallelConfig":
        """Return a normalized copy with minimum values enforced."""
        return ParallelConfig(
            tensor_parallel_size=max(1, self.tensor_parallel_size),
            pipeline_parallel_size=max(1, self.pipeline_parallel_size),
        )


def _normalize_parallel_config(
    config: Optional[Union["ParallelConfig", int, dict]]
) -> ParallelConfig:
    """Coerce user-provided config values into a ParallelConfig instance."""
    if config is None:
        return ParallelConfig()
    if isinstance(config, ParallelConfig):
        return config.with_defaults()
    if isinstance(config, int):
        return ParallelConfig(tensor_parallel_size=max(1, config))
    if isinstance(config, dict):
        tp = max(1, int(config.get("tensor_parallel_size", 1)))
        pp = max(1, int(config.get("pipeline_parallel_size", 1)))
        return ParallelConfig(tensor_parallel_size=tp, pipeline_parallel_size=pp)
    raise TypeError(
        f"Unsupported parallel_config type: {type(config)!r}. "
        "Expected ParallelConfig, int, dict, or None."
    )


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
            parallel_config: Optional tensor/pipeline parallel configuration
            block_size: Number of tokens per memory block
        """
        self.model_params = model_params
        self.parallel_config = _normalize_parallel_config(parallel_config)
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
        tp_size = max(1, self.parallel_config.tensor_parallel_size)
        per_shard_memory = self.hardware_params["vmemory"]
        total_memory = per_shard_memory * tp_size

        # memory for weights
        w_memory_total = self.get_weights_memory()
        w_memory_per_shard = w_memory_total / tp_size

        # Check if weights exceed per-shard memory
        if w_memory_per_shard >= per_shard_memory:
            print(
                "Warning: Model weights per shard "
                f"({naturalsize(w_memory_per_shard)}) exceed GPU memory "
                f"({naturalsize(per_shard_memory)}). No KV cache blocks available "
                f"even with tensor parallel size {tp_size}."
            )
            return 0

        num_kv_heads = llama_config.get_num_key_value_heads(self.model_params)
        heads_per_shard = max(1, math.ceil(num_kv_heads / tp_size))

        block_memory_size_per_shard = (
            2
            * self.block_size
            * heads_per_shard
            * llama_config.get_head_dim(self.model_params)
            * self.kv_bit
            / 8
        )
        total_block_memory_size_per_shard = (
            block_memory_size_per_shard
            * llama_config.get_num_hidden_layers(self.model_params)
        )

        available_memory_per_shard = per_shard_memory - w_memory_per_shard
        if available_memory_per_shard <= 0:
            return 0

        max_blocks_per_shard = math.floor(
            available_memory_per_shard / total_block_memory_size_per_shard
        )

        # Ensure we don't return negative values
        return max(0, max_blocks_per_shard)

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
        tp_size = max(1, self.parallel_config.tensor_parallel_size)
        per_shard_memory = self.hardware_params["vmemory"]
        weights_per_shard = self.get_weights_memory() / tp_size
        print(
            "Weights memory per shard / GPU memory: "
            f"{naturalsize(weights_per_shard)} / {naturalsize(per_shard_memory)} "
            f"(tensor_parallel_size={tp_size})"
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
            if num_blocks == 0:
                return
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

    def usage(self) -> tuple[int, int]:
        """Return the current (used, total) block counts."""
        return self._allocated_blocks, self._max_num_blocks
