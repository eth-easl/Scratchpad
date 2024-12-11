from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from typing import Union
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
from scratchpad.nn.layers.logits_processor import LogitsProcessor, LogitsMetadata
from scratchpad.distributed.communication_op import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from scratchpad.distributed.parallel_state import get_tensor_model_parallel_rank
from scratchpad.distributed.utils import split_tensor_along_last_dim
from scratchpad.model_executor.forward_info import ForwardBatch, ForwardMode
from triteia.python.ops import ldmm as ldmm


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

    def set_topping_info(self, bs, weight_indices, lora_buffer=None, delta_buffer=None):
        self.weight_indices = weight_indices
        self.delta_weights = delta_buffer

    def forward(self, input_: torch.Tensor):
        if self.delta_weights == None or self.delta_weights.shape[0] == 0:
            return self.base_layer(input_)

        base_output = torch.zeros(
            (input_.shape[0], self.delta_weights[0].shape[1]),
            device=input_.device,
            dtype=self.delta_weights[0].dtype,
        )
        unique_indices = torch.unique(self.weight_indices)

        for id in unique_indices:
            idx_mask = self.weight_indices == id
            inp = input_[idx_mask]
            if id == -1:
                w = self.base_layer.weight
            else:
                w = self.delta_weights[id]
            base_output[idx_mask] = nn.functional.embedding(inp, w)
        return base_output


class ColumnParallelLinearWithTopping(BaseLayerWithTopping):
    def __init__(self, base_layer: ColumnParallelLinear, config) -> None:
        super().__init__(base_layer, config)

    def set_topping_info(self, bs, weight_indices, lora_buffer=None, delta_buffer=None):
        self.weight_indices = weight_indices
        self.bs = bs
        if lora_buffer != None:
            self.A_buffer = lora_buffer[0]
            self.B_buffer = lora_buffer[1]
        else:
            self.A_buffer = torch.zeros(0, 0, 0)
            self.B_buffer = torch.zeros(0, 0, 0)

        if delta_buffer != None:
            self.qweight_buffer = delta_buffer[0]
            self.metas_buffer = delta_buffer[1]
            self.scales_buffer = delta_buffer[2]
        else:
            self.qweight_buffer = torch.zeros(0, 0, 0)
            self.metas_buffer = torch.zeros(0, 0, 0)
            self.scales_buffer = torch.zeros(0, 0, 0)

    def forward(self, input_: torch.Tensor):
        base_output = self.base_layer(input_)[0]
        output = ldmm(
            indices=self.weight_indices,
            x=input_,
            LwA=self.A_buffer,
            LwB=self.B_buffer,
            DeltaW=self.qweight_buffer,
            metas=self.metas_buffer,
            ss=self.scales_buffer,
        )
        base_output += output
        return base_output, None


class MergedColumnParallelLinearWithTopping(ColumnParallelLinearWithTopping):
    def __init__(self, base_layer: MergedColumnParallelLinear, config: Dict) -> None:
        super().__init__(base_layer, config)

    def set_topping_info(self, bs, weight_indices, lora_buffer=None, delta_buffer=None):
        self.weight_indices = weight_indices
        self.bs = bs
        if lora_buffer != None:
            self.A_buffer = lora_buffer[0]
            self.B_buffer = lora_buffer[1]
        else:
            self.A_buffer = torch.zeros(0, 0, 0)
            self.B_buffer = torch.zeros(0, 0, 0)

        if delta_buffer != None:
            self.qweight_buffer = delta_buffer[0]
            self.metas_buffer = delta_buffer[1]
            self.scales_buffer = delta_buffer[2]
        else:
            self.qweight_buffer = torch.zeros(0, 0, 0)
            self.metas_buffer = torch.zeros(0, 0, 0)
            self.scales_buffer = torch.zeros(0, 0, 0)

    def forward(self, input_: torch.Tensor):
        base_output = self.base_layer(input_)[0]
        rank = self.A_buffer.shape[2] // 2
        b_dim = self.B_buffer.shape[2] // 2
        qweight_dim = self.qweight_buffer.shape[2] // 2
        metas_dim = self.metas_buffer.shape[1] // 2
        scales_dim = self.scales_buffer.shape[2] // 2
        for i in range(2):
            output = ldmm(
                indices=self.weight_indices,
                x=input_,
                LwA=self.A_buffer[:, :, i * rank : (i + 1) * rank],
                LwB=self.B_buffer[:, :, i * b_dim : (i + 1) * b_dim],
                DeltaW=self.qweight_buffer[
                    :, :, i * qweight_dim : (i + 1) * qweight_dim
                ],
                metas=self.metas_buffer[:, i * metas_dim : (i + 1) * metas_dim, :],
                ss=self.scales_buffer[:, :, i * scales_dim : (i + 1) * scales_dim],
            )
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
        self,
        bs,
        weight_indices,
        lora_buffer=None,
        delta_buffer_q=None,
        delta_buffer_kv=None,
    ):
        self.set_lora = True
        self.bs = bs
        self.weight_indices = weight_indices
        if lora_buffer != None:
            self.A_buffer_qkv = lora_buffer[0]
            self.B_buffer_q = lora_buffer[1]
            self.B_buffer_kv = lora_buffer[2]
        else:
            self.A_buffer_qkv = torch.zeros(0, 0, 0)
            self.B_buffer_q = torch.zeros(0, 0, 0)
            self.B_buffer_kv = torch.zeros(0, 0, 0)
        if delta_buffer_q != None:
            self.qweight_buffer_q = delta_buffer_q[0]
            self.meta_buffer_q = delta_buffer_q[1]
            self.scales_buffer_q = delta_buffer_q[2]
        else:
            self.qweight_buffer_q = torch.zeros(0, 0, 0)
            self.meta_buffer_q = torch.zeros(0, 0, 0)
            self.scales_buffer_q = torch.zeros(0, 0, 0)

        if delta_buffer_kv != None:
            self.qweight_buffer_kv = delta_buffer_kv[0]
            self.meta_buffer_kv = delta_buffer_kv[1]
            self.scales_buffer_kv = delta_buffer_kv[2]
        else:
            self.qweight_buffer_kv = torch.zeros(0, 0, 0)
            self.meta_buffer_kv = torch.zeros(0, 0, 0)
            self.scales_buffer_kv = torch.zeros(0, 0, 0)

        # q,k,v have the same input dimensions
        # k,v have the same output dimensions
        # q has a different output dimension than k,v

        # (A_buffer_qkv: bsz, dim1, rank*2)
        # (B_buffer_q: bsz, rank, dim2*2)
        # (B_buffer_kv: bsz, rank, dim3*2)

        # (qweight_buffer: bsz,_, _*3)
        # (meta_buffer: bsz,_, _*3)
        # (scales_buffer: bsz, _, _*3)

    def forward(self, input_: torch.Tensor):
        base_output = self.base_layer(input_)[0]
        rank = self.A_buffer_qkv.shape[2] // 3
        b_dim_q = self.B_buffer_q.shape[2]
        b_dim_kv = self.B_buffer_kv.shape[2] // 2
        qweight_kv_dim = self.qweight_buffer_kv.shape[2] // 2
        metas_kv_dim = self.meta_buffer_kv.shape[1] // 2
        scales_kv_dim = self.scales_buffer_kv.shape[2] // 2

        for i in range(3):  # calculate q, k, v projections
            if i == 0:  # q
                output = ldmm(
                    indices=self.weight_indices,
                    x=input_,
                    LwA=self.A_buffer_qkv[:, :, i * rank : (i + 1) * rank],
                    LwB=self.B_buffer_q,
                    DeltaW=self.qweight_buffer_q,
                    metas=self.meta_buffer_q,
                    ss=self.scales_buffer_q,
                )
                base_output[:, i * b_dim_q : (i + 1) * b_dim_q] += output
            else:  # k, v
                output = ldmm(
                    indices=self.weight_indices,
                    x=input_,
                    LwA=self.A_buffer_qkv[:, :, i * rank : (i + 1) * rank],
                    LwB=self.B_buffer_kv[:, :, (i - 1) * b_dim_kv : i * b_dim_kv],
                    DeltaW=self.qweight_buffer_kv[
                        :, :, (i - 1) * qweight_kv_dim : i * qweight_kv_dim
                    ],
                    metas=self.meta_buffer_kv[
                        :, (i - 1) * metas_kv_dim : i * metas_kv_dim, :
                    ],
                    ss=self.scales_buffer_kv[
                        :, :, (i - 1) * scales_kv_dim : i * scales_kv_dim
                    ],
                )
                base_output[:, i * b_dim_kv : (i + 1) * b_dim_kv] += output

        return base_output, None


class RowParallelLinearWithTopping(BaseLayerWithTopping):
    def __init__(self, base_layer: RowParallelLinear, config: Dict) -> None:
        super().__init__(base_layer, config)

    def set_topping_info(self, bs, weight_indices, lora_buffer=None, delta_buffer=None):
        self.weight_indices = weight_indices
        self.bs = bs
        if lora_buffer != None:
            self.A_buffer = lora_buffer[0]
            self.B_buffer = lora_buffer[1]
        else:
            self.A_buffer = torch.zeros(0, 0, 0)
            self.B_buffer = torch.zeros(0, 0, 0)

        if delta_buffer != None:
            self.qweight_buffer = delta_buffer[0]
            self.metas_buffer = delta_buffer[1]
            self.scales_buffer = delta_buffer[2]
        else:
            self.qweight_buffer = torch.zeros(0, 0, 0)
            self.metas_buffer = torch.zeros(0, 0, 0)
            self.scales_buffer = torch.zeros(0, 0, 0)

    def forward(self, input_: torch.Tensor):
        base_output = torch.matmul(input_, self.base_layer.weight.T)
        delta_output = ldmm(
            indices=self.weight_indices,
            x=input_,
            LwA=self.A_buffer,
            LwB=self.B_buffer,
            DeltaW=self.qweight_buffer,
            metas=self.metas_buffer,
            ss=self.scales_buffer,
        )
        print(f"weight_indices: {self.weight_indices}")
        print(f"A_buffer.shape: {self.A_buffer.shape}")
        print(f"base_output.shape: {base_output.shape}")
        print(f"delta_output.shape: {delta_output.shape}")
        print(f"base: {base_output}")
        print(f"max delta: {torch.max(abs(delta_output))}")
        assert base_output.shape == delta_output.shape
        output_ = base_output + delta_output
        # output_ = base_output
        if not self.base_layer.skip_bias_add:
            output = (
                output_ + self.base_layer.bias
                if self.base_layer.bias is not None
                else output_
            )
            output_bias = None
        else:
            output = output_
            output_bias = self.base_layer.bias
        return output, output_bias


class LogitsProcessorWithTopping(BaseLayerWithTopping):
    def __init__(self, base_layer, config):
        super().__init__(base_layer, config)

    def set_topping_info(self, bs, weight_indices, lora_buffer=None, delta_buffer=None):
        self.weight_indices = weight_indices
        self.delta_buffer = delta_buffer

    def _get_logits(
        self,
        input_ids,
        hidden_states,
        weight,
        logits_metadata: Union[LogitsMetadata, ForwardBatch],
    ):
        if self.delta_buffer == None or self.delta_buffer.shape[0] == 0:
            return self.base_layer._get_logits(
                input_ids, hidden_states, weight, logits_metadata
            )
        if isinstance(logits_metadata, ForwardBatch):
            logits_metadata = LogitsMetadata.from_forward_batch(logits_metadata)
        assert isinstance(logits_metadata, LogitsMetadata)

        # Get the last hidden states and last logits for the next token prediction
        if logits_metadata.forward_mode.is_decode():
            last_index = None
            last_hidden = hidden_states
            base_output = torch.zeros(
                last_hidden.shape[0],
                self.delta_buffer[0].shape[0],
                dtype=last_hidden.dtype,
                device=last_hidden.device,
            )
            unique_indices = torch.unique(self.weight_indices)
            for id in unique_indices:
                idx_mask = self.weight_indices == id
                inp = last_hidden[idx_mask]
                if id == -1:
                    w = weight.T
                else:
                    w = self.delta_buffer[id]
                output = nn.functional.linear(inp, w)
                base_output[idx_mask] = output
        else:
            last_index = torch.cumsum(logits_metadata.extend_seq_lens, dim=0) - 1
            last_hidden = hidden_states[last_index]
            base_output = torch.zeros(
                last_hidden.shape[0],
                self.delta_buffer[0].shape[0],
                dtype=last_hidden.dtype,
                device=last_hidden.device,
            )
            unique_indices = torch.unique(self.weight_indices)
            assert len(unique_indices) == 1, f"Prefill stage only supports one index"
            w_idx = unique_indices[0]
            if w_idx == -1:
                w = weight.T
            else:
                w = self.delta_buffer[w_idx]
            output = nn.functional.linear(last_hidden, w)
            base_output = output
        return base_output

    def forward(
        self,
        input_ids,
        hidden_states,
        weight,
        logits_metadata: Union[LogitsMetadata, ForwardBatch],
        last_logits: Optional[torch.Tensor] = None,
    ):
        return self.base_layer(
            input_ids,
            hidden_states,
            weight,
            logits_metadata,
            self._get_logits(input_ids, hidden_states, weight, logits_metadata),
        )


def get_topping_layer(layer: nn.Module) -> BaseLayerWithTopping:
    supported_layer_types = {
        # the order matters
        VocabParallelEmbedding: VocabParallelEmbeddingWithTopping,
        QKVParallelLinear: QKVParallelLinearWithToppings,
        MergedColumnParallelLinear: MergedColumnParallelLinearWithTopping,
        ColumnParallelLinear: ColumnParallelLinearWithTopping,
        RowParallelLinear: RowParallelLinearWithTopping,
        LogitsProcessor: LogitsProcessorWithTopping,
    }
    for src_layer_type, topping_layer_type in supported_layer_types.items():
        if isinstance(layer, src_layer_type):  # pylint: disable=unidiomatic-typecheck
            ret = topping_layer_type(layer, {})
            return ret
    raise Exception(f"No corresponding Topping layer supported for {type(layer)}.")
