from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn

from scratchpad.server.args import ServerArgs
from scratchpad.nn.utils import update_flashinfer_indices
from scratchpad.server.args import global_args
from scratchpad.model_executor.forward_info import ForwardMode, ForwardBatch


if TYPE_CHECKING:
    from scratchpad.model_executor.model_runner import ModelRunner
    from .radix_attention import RadixAttention


class AttentionBackend(ABC):
    """The base class of attention backends"""

    @abstractmethod
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        raise NotImplementedError()

    def init_cuda_graph_state(self, max_bs: int):
        """Init the global shared states for cuda graph."""
        raise NotImplementedError()

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices,
        seq_lens,
        encoder_lens: Optional[torch.Tensor] = None,
    ):
        """Init the metadata for a forward pass for capturing a cuda graph."""
        raise NotImplementedError()

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices,
        seq_lens,
        encoder_lens: Optional[torch.Tensor] = None,
    ):
        """Init the metadata for a forward pass for replying a cuda graph."""
        raise NotImplementedError()

    def get_cuda_graph_seq_len_fill_value(self):
        """Get the fill value for padded seq lens. Typically, it is 0 or 1."""
        raise NotImplementedError()

    def forward(self, q, k, v, layer: "RadixAttention", forward_batch: ForwardBatch):
        """Run forward on an attention layer."""
        if forward_batch.forward_mode.is_decode():
            return self.forward_decode(q, k, v, layer, forward_batch)
        else:
            return self.forward_extend(q, k, v, layer, forward_batch)

    def forward_decode(
        self, q, k, v, layer: "RadixAttention", forward_batch: ForwardBatch
    ):
        """Run a forward for decode."""
        raise NotImplementedError()

    def forward_extend(
        self, q, k, v, layer: "RadixAttention", forward_batch: ForwardBatch
    ):
        """Run a forward for extend."""
        raise NotImplementedError()


class TritonAttnBackend(AttentionBackend):
    def __init__(self, model_runner: "ModelRunner"):
        # Lazy import to avoid the initialization of cuda context
        from .triton_attn.decode_attention import (
            decode_attention_fwd,
        )
        from .triton_attn.extend_attention import (
            extend_attention_fwd,
        )

        super().__init__()

        self.decode_attention_fwd = decode_attention_fwd
        self.extend_attention_fwd = extend_attention_fwd
        self.num_head = (
            model_runner.model_config.num_attention_heads // model_runner.tp_size
        )

        if global_args.triton_attention_reduce_in_fp32:
            self.reduce_dtype = torch.float32
        else:
            self.reduce_dtype = torch.float16

        self.forward_metadata = None

        self.cuda_graph_max_seq_len = model_runner.model_config.context_len

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init auxiliary variables for triton attention backend."""

        if forward_batch.forward_mode.is_decode():
            start_loc = torch.zeros_like(forward_batch.seq_lens, dtype=torch.int32)
            start_loc[1:] = torch.cumsum(forward_batch.seq_lens[:-1], dim=0)

            total_num_tokens = torch.sum(forward_batch.seq_lens).item()
            attn_logits = torch.empty(
                (self.num_head, total_num_tokens),
                dtype=self.reduce_dtype,
                device="cuda",
            )

            max_seq_len = torch.max(forward_batch.seq_lens).item()
            max_extend_len = None
        else:
            start_loc = attn_logits = max_seq_len = None
            prefix_lens = forward_batch.extend_prefix_lens
            max_extend_len = torch.max(forward_batch.seq_lens - prefix_lens).item()

        self.forward_metadata = start_loc, attn_logits, max_seq_len, max_extend_len

    def init_cuda_graph_state(self, max_bs: int):
        self.cuda_graph_max_total_num_tokens = max_bs * self.cuda_graph_max_seq_len

        self.cuda_graph_start_loc = torch.zeros(
            (max_bs,), dtype=torch.int32, device="cuda"
        )
        self.cuda_graph_attn_logits = torch.empty(
            (
                self.num_head,
                self.cuda_graph_max_total_num_tokens,
            ),
            dtype=self.reduce_dtype,
            device="cuda",
        )

    def init_forward_metadata_capture_cuda_graph(
        self, bs: int, req_pool_indices, seq_lens
    ):
        self.forward_metadata = (
            self.cuda_graph_start_loc,
            self.cuda_graph_attn_logits,
            self.cuda_graph_max_seq_len,
            None,
        )

    def init_forward_metadata_replay_cuda_graph(
        self, bs: int, req_pool_indices, seq_lens
    ):
        self.cuda_graph_start_loc.zero_()
        self.cuda_graph_start_loc[1:bs] = torch.cumsum(seq_lens[: bs - 1], dim=0)

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def forward_extend(self, q, k, v, layer: nn.Module, forward_batch: ForwardBatch):
        # TODO: reuse the buffer across layers
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        forward_batch.token_to_kv_pool.set_kv_buffer(
            layer.layer_id, forward_batch.out_cache_loc, k, v
        )

        start_loc, attn_logits, max_seq_len, max_extend_len = self.forward_metadata
        self.extend_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k.contiguous(),
            v.contiguous(),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.extend_seq_lens,
            forward_batch.extend_start_loc,
            max_extend_len,
            layer.scaling,
            layer.logit_cap,
        )
        return o

    def forward_decode(self, q, k, v, layer: nn.Module, forward_batch: ForwardBatch):
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        # TODO: reuse the buffer across layers
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        start_loc, attn_logits, max_seq_len, max_extend_len = self.forward_metadata

        forward_batch.token_to_kv_pool.set_kv_buffer(
            layer.layer_id, forward_batch.out_cache_loc, k, v
        )

        self.decode_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            start_loc,
            forward_batch.seq_lens,
            attn_logits,
            max_seq_len,
            layer.scaling,
            layer.logit_cap,
        )
        return o
