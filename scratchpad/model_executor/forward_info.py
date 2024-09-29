from enum import IntEnum, auto
from dataclasses import dataclass
import torch
import numpy as np
from typing import List, TYPE_CHECKING
from scratchpad.memory.pool import ReqToTokenPool, BaseTokenToKVPool

if TYPE_CHECKING:
    from scratchpad.nn.attention.backend import AttentionBackend
    from scratchpad.scheduler.schedule_batch import ScheduleBatch
    from scratchpad.model_executor.model_runner import ModelRunner


@dataclass
class ForwardMode(IntEnum):
    # Extend a sequence. The KV cache of the first part of the sequence is already computed (e.g., system prompt).

    EXTEND = auto()
    # Decode one token.
    DECODE = auto()
    # Contains both PREFILL and EXTEND.
    MIXED = auto()

    def is_extend(self):
        # print(f"self {self}, {ForwardMode.EXTEND}")
        # print(f"self == ForwardMode.EXTEND {self == 1}")
        # (note: xiaozhe) fix later: change 1 to ForwardMode.EXTEND, etc.
        return self == 1 or self == 3

    def is_decode(self):
        return self == 2

    def is_mixed(self):
        return self == 3


@dataclass
class InputMetadata:
    """Store all inforamtion of a forward pass."""

    forward_mode: ForwardMode
    batch_size: int
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool: BaseTokenToKVPool
    attn_backend: "AttentionBackend"

    # Output location of the KV cache
    out_cache_loc: torch.Tensor

    # Position information
    positions: torch.Tensor = None

    # For extend
    extend_seq_lens: torch.Tensor = None
    extend_prefix_lens: torch.Tensor = None
    extend_start_loc: torch.Tensor = None
    extend_no_prefix: bool = None

    # For logprob
    return_logprob: bool = False
    top_logprobs_nums: List[int] = None
    extend_seq_lens_cpu: List[int] = None
    extend_logprob_start_lens_cpu: List[int] = None

    # For multimodal
    pixel_values: List[torch.Tensor] = None
    image_sizes: List[List[List[int]]] = None
    image_offsets: List[List[int]] = None
    modalities: List[List[str]] = None

    def init_multimuldal_info(self, batch: "ScheduleBatch"):
        reqs = batch.reqs
        self.pixel_values = [r.pixel_values for r in reqs]
        self.image_sizes = [r.image_sizes for r in reqs]
        self.image_offsets = [r.image_offsets for r in reqs]
        self.modalities = [r.modalities for r in reqs]

    def compute_positions(self, batch: "ScheduleBatch"):
        if self.forward_mode.is_decode():

            self.positions = self.seq_lens - 1

        else:
            self.positions = torch.tensor(
                np.concatenate(
                    [
                        np.arange(batch.prefix_lens_cpu[i], len(req.fill_ids))
                        for i, req in enumerate(batch.reqs)
                    ],
                    axis=0,
                ),
                device="cuda",
            )

        # Positions should be in long type
        self.positions = self.positions.to(torch.int64)

    def compute_extend_infos(self, batch: "ScheduleBatch"):
        self.extend_seq_lens = torch.tensor(batch.extend_lens_cpu, device="cuda")
        self.extend_prefix_lens = torch.tensor(batch.prefix_lens_cpu, device="cuda")
        self.extend_start_loc = torch.zeros_like(self.extend_seq_lens)
        self.extend_start_loc[1:] = torch.cumsum(self.extend_seq_lens[:-1], dim=0)
        self.extend_no_prefix = all(x == 0 for x in batch.prefix_lens_cpu)
        self.extend_seq_lens_cpu = batch.extend_lens_cpu
        self.extend_logprob_start_lens_cpu = batch.extend_logprob_start_lens_cpu

    @classmethod
    def from_schedule_batch(
        cls,
        model_runner: "ModelRunner",
        batch: "ScheduleBatch",
    ):
        ret = cls(
            forward_mode=batch.forward_mode,
            batch_size=batch.batch_size(),
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            req_to_token_pool=model_runner.req_to_token_pool,
            token_to_kv_pool=model_runner.token_to_kv_pool,
            attn_backend=model_runner.attn_backend,
            out_cache_loc=batch.out_cache_loc,
            return_logprob=batch.return_logprob,
            top_logprobs_nums=batch.top_logprobs_nums,
        )

        ret.compute_positions(batch)

        if not batch.forward_mode.is_decode():
            ret.init_multimuldal_info(batch)
            ret.compute_extend_infos(batch)

        model_runner.attn_backend.init_forward_metadata(batch, ret)

        return ret
