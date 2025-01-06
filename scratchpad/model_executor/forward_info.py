from enum import IntEnum, auto
from dataclasses import dataclass
import torch
import numpy as np
from typing import List, TYPE_CHECKING, Optional

from scratchpad.memory.pool import ReqToTokenPool, BaseTokenToKVPool
from scratchpad.nn.layers.rotary_embedding import MRotaryEmbedding

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

    # The sum of all sequence lengths
    seq_lens_sum: int

    # For logprob
    return_logprob: bool = False
    top_logprobs_nums: Optional[List[int]] = None

    # Position information
    positions: torch.Tensor = None

    # For extend
    extend_num_tokens: Optional[int] = None
    extend_seq_lens: Optional[torch.Tensor] = None
    extend_prefix_lens: Optional[torch.Tensor] = None
    extend_start_loc: Optional[torch.Tensor] = None
    extend_seq_lens_cpu: Optional[List[int]] = None
    extend_logprob_start_lens_cpu: Optional[List[int]] = None

    # For multimodal
    image_inputs: Optional[List["ImageInputs"]] = None

    # Encoder-decoder
    encoder_cached: Optional[List[bool]] = None
    encoder_lens: Optional[torch.Tensor] = None
    encoder_lens_cpu: Optional[List[int]] = None
    encoder_out_cache_loc: Optional[torch.Tensor] = None

    # For Toppings
    topping_paths: Optional[List[str]] = None

    # Sampling info
    sampling_info: "SamplingBatchInfo" = None

    # Attention backend
    req_to_token_pool: ReqToTokenPool = None
    token_to_kv_pool: BaseTokenToKVPool = None
    attn_backend: "AttentionBackend" = None

    # For Qwen2-VL
    mrope_positions: torch.Tensor = None

    def compute_mrope_positions(
        self, model_runner: "ModelRunner", batch: "ModelWorkerBatch"
    ):
        device = model_runner.device
        hf_config = model_runner.model_config.hf_config
        mrope_positions_list = [None] * self.seq_lens.shape[0]
        if self.forward_mode.is_decode():
            for i, _ in enumerate(mrope_positions_list):
                mrope_positions_list[i] = MRotaryEmbedding.get_next_input_positions(
                    batch.mrope_positions_delta[i][0],
                    int(self.seq_lens[i]) - 1,
                    int(self.seq_lens[i]),
                )
        elif self.forward_mode.is_extend():
            extend_start_loc_cpu = self.extend_start_loc.cpu().numpy()
            for i, image_inputs in enumerate(batch.image_inputs):
                extend_start_loc, extend_seq_len, extend_prefix_len = (
                    extend_start_loc_cpu[i],
                    batch.extend_seq_lens[i],
                    batch.extend_prefix_lens[i],
                )
                if image_inputs is None:
                    # text only
                    mrope_positions = [
                        [
                            pos
                            for pos in range(
                                extend_prefix_len, extend_prefix_len + extend_seq_len
                            )
                        ]
                    ] * 3
                    mrope_position_delta = 0
                else:
                    # TODO: current qwen2-vl do not support radix cache since mrope position calculation
                    (
                        mrope_positions,
                        mrope_position_delta,
                    ) = MRotaryEmbedding.get_input_positions(
                        input_tokens=self.input_ids[
                            extend_start_loc : extend_start_loc + extend_seq_len
                        ],
                        image_grid_thw=image_inputs.image_grid_thws,
                        vision_start_token_id=hf_config.vision_start_token_id,
                        spatial_merge_size=hf_config.vision_config.spatial_merge_size,
                        context_len=0,
                    )
                mrope_positions_list[i] = mrope_positions
                batch.mrope_positions_delta[i].append(mrope_position_delta)

        self.mrope_positions = torch.concat(
            [torch.tensor(pos, device=device) for pos in mrope_positions_list],
            axis=1,
        )
        self.mrope_positions = self.mrope_positions.to(torch.int64)

    @classmethod
    def init_new(
        cls,
        batch: "ModelWorkerBatch",
        model_runner: "ModelRunner",
    ):

        device = model_runner.device
        ret = cls(
            forward_mode=batch.forward_mode,
            batch_size=len(batch.seq_lens),
            input_ids=batch.input_ids,
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            out_cache_loc=batch.out_cache_loc,
            image_inputs=batch.image_inputs,
            encoder_cached=batch.encoder_cached,
            encoder_lens=batch.encoder_lens,
            encoder_lens_cpu=batch.encoder_lens_cpu,
            encoder_out_cache_loc=batch.encoder_out_cache_loc,
            seq_lens_sum=batch.seq_lens_sum,
            return_logprob=batch.return_logprob,
            top_logprobs_nums=batch.top_logprobs_nums,
            topping_paths=batch.topping_paths,
            sampling_info=batch.sampling_info,
        )

        # Init position information
        if not ret.forward_mode.is_decode():
            ret.positions = torch.concat(
                [
                    torch.arange(prefix_len, prefix_len + extend_len, device=device)
                    for prefix_len, extend_len in zip(
                        batch.extend_prefix_lens, batch.extend_seq_lens
                    )
                ],
                axis=0,
            )
            ret.extend_num_tokens = batch.extend_num_tokens
            ret.extend_seq_lens = torch.tensor(
                batch.extend_seq_lens, dtype=torch.int32
            ).to(device, non_blocking=True)

            ret.extend_prefix_lens = torch.tensor(
                batch.extend_prefix_lens, dtype=torch.int32
            ).to(device, non_blocking=True)
            ret.extend_start_loc = torch.zeros_like(ret.extend_seq_lens)
            ret.extend_start_loc[1:] = torch.cumsum(ret.extend_seq_lens[:-1], dim=0)
            ret.extend_seq_lens_cpu = batch.extend_seq_lens
            ret.extend_logprob_start_lens_cpu = batch.extend_logprob_start_lens

        if model_runner.model_is_mrope:
            ret.compute_mrope_positions(model_runner, batch)

        # Init attention information
        ret.req_to_token_pool = model_runner.req_to_token_pool
        ret.token_to_kv_pool = model_runner.token_to_kv_pool
        ret.attn_backend = model_runner.attn_backend

        # Init lora information
        if model_runner.server_args.enable_toppings:
            model_runner.topping_manager.prepare_topping_batch(ret)
        return ret
