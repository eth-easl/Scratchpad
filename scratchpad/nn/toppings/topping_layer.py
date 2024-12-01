import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from scratchpad.nn.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from scratchpad.nn.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from scratchpad.distributed.communication_op import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from scratchpad.distributed.parallel_state import get_tensor_model_parallel_rank
from scratchpad.distributed.utils import split_tensor_along_last_dim
from scratchpad.model_executor.forward_info import ForwardBatch, ForwardMode


class BaseLayerWithTopping(nn.Module):
    def __init__(self, base_layer, config: Dict):
        super().__init__()
        self.base_layer = base_layer
        self.config = config
        self.set_topping = False

    def forward(self, x: torch.Tensor):
        return self.base_layer.forward(x)

    def set_topping_info(self, *args):
        pass


class VocabParallelEmbeddingWithTopping(BaseLayerWithTopping):
    def __init__(self, base_layer: VocabParallelEmbedding, config: Dict) -> None:
        super().__init__(base_layer, config)
        self.weight = base_layer.weight


class ColumnParallelLinearWithTopping(BaseLayerWithTopping):
    def __init__(self, base_layer: ColumnParallelLinear, config) -> None:
        super().__init__(base_layer, config)

    def apply_topping(self, output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # TODO
        raise NotImplementedError("apply_topping method is not implemented yet")

    def forward(self, input_: torch.Tensor):
        # duplicate the logic in ColumnParallelLinear
        raise NotImplementedError("The forward method is not implemented yet.")


class MergedColumnParallelLinearWithTopping(ColumnParallelLinearWithTopping):
    def __init__(self, base_layer: MergedColumnParallelLinear, config: Dict) -> None:
        super().__init__(base_layer, config)


class QKVParallelLinearWithToppings(ColumnParallelLinearWithTopping):
    def __init__(
        self,
        base_layer: QKVParallelLinear,
        config: Dict,
    ) -> None:
        super().__init__(base_layer, config)

    def set_lora_info(
        self, A_buffer_qkv, B_buffer_q, B_buffer_kv, bs, seg_indptr, weight_indices
    ):
        self.set_lora = True
        self.A_buffer_qkv = A_buffer_qkv
        self.B_buffer_q = B_buffer_q
        self.B_buffer_kv = B_buffer_kv
        self.bs = bs
        self.seg_indptr = seg_indptr
        self.weight_indices = weight_indices

    def apply_topping(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        pass


class RowParallelLinearWithTopping(BaseLayerWithTopping):
    def __init__(self, base_layer: RowParallelLinear, config: Dict) -> None:
        super().__init__(base_layer, config)

    def set_lora_info(self, A_buffer, B_buffer, bs, seg_indptr, weight_indices):
        self.set_lora = True
        self.A_buffer = A_buffer
        self.B_buffer = B_buffer
        self.bs = bs
        self.seg_indptr = seg_indptr
        self.weight_indices = weight_indices

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("apply_lora method is not implemented yet")

    def forward(self, input_):
        raise NotImplementedError("The forward method needs to be implemented.")


def get_topping_layer(
    layer: nn.Module, segment_gemm, lora_rank, scaling
) -> BaseLayerWithTopping:
    supported_layer_types = {
        # the order matters
        VocabParallelEmbedding: VocabParallelEmbeddingWithTopping,
        QKVParallelLinear: QKVParallelLinearWithToppings,
        MergedColumnParallelLinear: MergedColumnParallelLinearWithTopping,
        ColumnParallelLinear: ColumnParallelLinearWithTopping,
        RowParallelLinear: RowParallelLinearWithTopping,
    }
    for src_layer_type, topping_layer_type in supported_layer_types.items():
        if isinstance(layer, src_layer_type):  # pylint: disable=unidiomatic-typecheck
            ret = topping_layer_type(layer, segment_gemm, lora_rank, scaling)
            return ret
    raise Exception(f"No corresponding Topping layer supported for {type(layer)}.")
