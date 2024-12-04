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

from triteia.python import ldmm


class BaseLayerWithTopping(nn.Module):
    def __init__(self, base_layer, config: Dict):
        super().__init__()
        self.base_layer = base_layer
        self.config = config
        self.set_topping = False

    def forward(self, x: torch.Tensor):
        return self.base_layer.forward(x)

    def set_topping_info(self, *args):
        ...


class VocabParallelEmbeddingWithTopping(BaseLayerWithTopping):
    def __init__(self, base_layer: VocabParallelEmbedding, config: Dict) -> None:
        super().__init__(base_layer, config)
        self.weight = base_layer.weight

    def forward(self, input_: torch.Tensor):
        return self.base_layer(input_)


class ColumnParallelLinearWithTopping(BaseLayerWithTopping):
    def __init__(self, base_layer: ColumnParallelLinear, config) -> None:
        super().__init__(base_layer, config)

    def set_topping_info(self, A_buffer, B_buffer, bs, weight_indices):
        pass

    def forward(self, input_: torch.Tensor):
        return self.base_layer(input_)


class MergedColumnParallelLinearWithTopping(ColumnParallelLinearWithTopping):
    def __init__(self, base_layer: MergedColumnParallelLinear, config: Dict) -> None:
        super().__init__(base_layer, config)

    def set_topping_info(self, A_buffer, B_buffer, bs, weight_indices, DeltaW_buffer, metas_buffer, ss_buffer):
        self.A_buffer = A_buffer
        self.B_buffer = B_buffer
        self.weight_indices = weight_indices
        self.bs = bs
        self.DeltaW = DeltaW_buffer
        self.metas = metas_buffer
        self.ss = ss_buffer
        # model.layers.24.mlp.gate_up_proj
        # (A_buffer: bsz, dim1, rank*2)
        # (B_buffer: bsz, rank, dim2*2)
        # (Deltaw: bsz,_, _*2)
        # (metas: bsz,_, _*2)
        # (ss: bsz, _, _*2)

    def forward(self, input_: torch.Tensor):
        # input_: (bsz, dim0)
        # indices: [0]
        # reshape indices such that it is (bsz, 1)
        base_output = self.base_layer(input_)[0]
        rank = self.A_buffer.shape[2] // 2
        b_dim = self.B_buffer.shape[2] // 2
        delta_dim = self.DeltaW.shape[2] // 2
        metas_dim = self.metas.shape[2] // 2
        ss_dim = self.ss.shape[2] // 2
        for i in range(2):
            output = ldmm(
                indices=self.weight_indices,
                x=input_,
                LwA=self.A_buffer[:, :, i * rank : (i + 1) * rank],
                LwB=self.B_buffer[:, :, i * b_dim : (i + 1) * b_dim],
                DeltaW=self.DeltaW[:, :, i * delta_dim : (i + 1) * delta_dim],
                metas=self.metas[:, :, i * metas_dim: (i + 1) * metas_dim],
                ss=self.ss[:, :, i * ss_dim: (i + 1) * ss_dim],
            )
            print(f"output={output.shape}")
            print(f"base_output={base_output.shape}")
            base_output[:, i * b_dim : (i + 1) * b_dim] += output
        return base_output, None


class QKVParallelLinearWithToppings(ColumnParallelLinearWithTopping):
    def __init__(
        self,
        base_layer: QKVParallelLinear,
        config: Dict,
    ) -> None:
        super().__init__(base_layer, config)

    def set_topping_info(
        self, A_buffer_qkv, B_buffer_q, B_buffer_kv, bs, weight_indices,
        DeltaW_buffer_kv, metas_buffer_kv, ss_buffer_kv,
        DeltaW_buffer_q, metas_buffer_q, ss_buffer_q
    ):
        self.set_lora = True
        self.A_buffer_qkv = A_buffer_qkv
        self.B_buffer_q = B_buffer_q
        self.B_buffer_kv = B_buffer_kv
        self.bs = bs
        self.weight_indices = weight_indices
        self.DeltaW_kv = DeltaW_buffer_kv
        self.metas_kv = metas_buffer_kv
        self.ss_kv = ss_buffer_kv
        self.DeltaW_q = DeltaW_buffer_q
        self.metas_q = metas_buffer_q
        self.ss_q = ss_buffer_q

        # q,k,v have the same input dimensions
        # k,v have the same output dimensions
        # q has a different output dimension than k,v
        
        # (A_buffer_qkv: bsz, dim1, rank*2)
        # (B_buffer_q: bsz, rank, dim2*2)
        # (B_buffer_kv: bsz, rank, dim3*2)

        # (DeltaW_kv: bsz,_, _*2)
        # (metas_kv: bsz,_, _*2)
        # (ss_kv: bsz, _, _*2)

        # (DeltaW_q: bsz,_, _)
        # (metas_q: bsz,_, _)
        # (ss_q: bsz, _, _)

    def apply_topping(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, input_: torch.Tensor):
        # input_: (bsz, dim0)
        # indices: [0]
        # reshape indices such that it is (bsz, 1)
        base_output = self.base_layer(input_)[0]
        rank = self.A_buffer_qkv.shape[2] // 2
        b_dim_q = self.B_buffer_q.shape[2]
        b_dim_kv = self.B_buffer_kv.shape[2] // 2
        delta_dim_kv = self.DeltaW_kv.shape[2] // 2
        metas_dim_kv = self.metas_kv.shape[2] // 2
        ss_dim_kv = self.ss_kv.shape[2] // 2

        for i in range(3): # calculate q, k, v projections
            if i == 0: # q
                output = ldmm(
                    indices=self.weight_indices,
                    x=input_,
                    LwA=self.A_buffer_qkv[:, :, i * rank : (i + 1) * rank],
                    LwB=self.B_buffer_q,
                    DeltaW=self.DeltaW_q,
                    metas=self.metas_q,
                    ss=self.ss_q,
                )
                base_output[:, i * b_dim_q : (i + 1) * b_dim_q] += output
            else: # k, v
                output = ldmm(
                    indices=self.weight_indices,
                    x=input_,
                    LwA=self.A_buffer_qkv[:, :, i * rank : (i + 1) * rank],
                    LwB=self.B_buffer_kv[:, :, (i - 1) * b_dim_kv : i * b_dim_kv],
                    DeltaW=self.DeltaW_kv[:, :, (i - 1) * delta_dim_kv : i * delta_dim_kv],
                    metas=self.metas_kv[:, :, (i - 1) * metas_dim_kv: i * metas_dim_kv],
                    ss=self.ss_kv[:, :, (i - 1) * ss_dim_kv: i * ss_dim_kv],
                )
                base_output[:, i * b_dim_kv : (i + 1) * b_dim_kv] += output               

        return base_output, None


class RowParallelLinearWithTopping(BaseLayerWithTopping):
    def __init__(self, base_layer: RowParallelLinear, config: Dict) -> None:
        super().__init__(base_layer, config)

    def set_topping_info(self, A_buffer, B_buffer, bs, weight_indices):
        self.set_lora = True
        self.A_buffer = A_buffer
        self.B_buffer = B_buffer
        self.bs = bs
        self.weight_indices = weight_indices

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("apply_lora method is not implemented yet")

    def forward(self, input_: torch.Tensor):
        return self.base_layer(input_)


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
            ret = topping_layer_type(layer, {})
            return ret
    raise Exception(f"No corresponding Topping layer supported for {type(layer)}.")
