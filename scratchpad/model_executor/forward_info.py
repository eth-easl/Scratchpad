from enum import IntEnum, auto
from dataclasses import dataclass
import torch
import numpy as np
from typing import List, TYPE_CHECKING, Optional

from scratchpad.memory.pool import ReqToTokenPool, BaseTokenToKVPool

if TYPE_CHECKING:
    from scratchpad.nn.attention.backend import AttentionBackend
    from scratchpad.scheduler.schedule_batch import ModelWorkerBatch, ImageInputs
    from scratchpad.model_executor.model_runner import ModelRunner
    from scratchpad.sampling.sampling_batch_info import SamplingBatchInfo


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
class ForwardBatch:
    """Store all inputs of a forward pass."""

    # The forward mode
    forward_mode: ForwardMode
    # The batch size
    batch_size: int
    # The input ids
    input_ids: torch.Tensor
    # The indices of requests in the req_to_token_pool
    req_pool_indices: torch.Tensor
    # The sequence length
    seq_lens: torch.Tensor
    # The indices of output tokens in the token_to_kv_pool
    out_cache_loc: torch.Tensor

    # For logprob
    return_logprob: bool = False
    top_logprobs_nums: Optional[List[int]] = None

    # Position information
    positions: torch.Tensor = None

    # For extend
    extend_seq_lens: Optional[torch.Tensor] = None
    extend_prefix_lens: Optional[torch.Tensor] = None
    extend_start_loc: Optional[torch.Tensor] = None
    extend_seq_lens_cpu: Optional[List[int]] = None
    extend_logprob_start_lens_cpu: Optional[List[int]] = None

    # For multimodal
    image_inputs: Optional[List["ImageInputs"]] = None

    # For LoRA
    lora_paths: Optional[List[str]] = None

    # Sampling info
    sampling_info: "SamplingBatchInfo" = None

    # Attention backend
    req_to_token_pool: ReqToTokenPool = None
    token_to_kv_pool: BaseTokenToKVPool = None
    attn_backend: "AttentionBackend" = None

    @classmethod
    def init_new(
        cls,
        batch: "ModelWorkerBatch",
        model_runner: "ModelRunner",
    ):
        device = "cuda"

        ret = cls(
            forward_mode=batch.forward_mode,
            batch_size=len(batch.seq_lens),
            input_ids=torch.tensor(batch.input_ids, dtype=torch.int32, device=device),
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            out_cache_loc=batch.out_cache_loc,
            return_logprob=batch.return_logprob,
            top_logprobs_nums=batch.top_logprobs_nums,
            lora_paths=batch.lora_paths,
            sampling_info=batch.sampling_info,
        )

        # Init position information
        if ret.forward_mode.is_decode():
            ret.positions = (ret.seq_lens - 1).to(torch.int64)
        else:
            ret.positions = torch.tensor(
                np.concatenate(
                    [
                        np.arange(prefix_len, prefix_len + extend_len)
                        for prefix_len, extend_len in zip(
                            batch.extend_prefix_lens, batch.extend_seq_lens
                        )
                    ],
                    axis=0,
                ),
                device=device,
            ).to(torch.int64)

            ret.image_inputs = batch.image_inputs
            ret.extend_seq_lens = torch.tensor(batch.extend_seq_lens, device=device)
            ret.extend_prefix_lens = torch.tensor(
                batch.extend_prefix_lens, device=device
            )
            ret.extend_start_loc = torch.zeros_like(ret.extend_seq_lens)
            ret.extend_start_loc[1:] = torch.cumsum(ret.extend_seq_lens[:-1], dim=0)
            ret.extend_seq_lens_cpu = batch.extend_seq_lens
            ret.extend_logprob_start_lens_cpu = batch.extend_logprob_start_lens

        # Init attention information
        ret.req_to_token_pool = model_runner.req_to_token_pool
        ret.token_to_kv_pool = model_runner.token_to_kv_pool
        ret.attn_backend = model_runner.attn_backend
        model_runner.attn_backend.init_forward_metadata(ret)

        # Init lora information
        if model_runner.server_args.lora_paths is not None:
            model_runner.lora_manager.prepare_lora_batch(ret)

        return ret
