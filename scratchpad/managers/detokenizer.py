import asyncio
import dataclasses
from typing import List, Dict, Union

import uvloop
import zmq
import zmq.asyncio
from collections import OrderedDict
from scratchpad.utils import get_tokenizer
from .structs import (
    BatchEmbeddingOut,
    BatchStrOut,
    BatchTokenIDOut,
    UpdateWeightReqOutput,
)
from scratchpad.scheduler.schedule_batch import FINISH_MATCHED_STR
from scratchpad.server.args import ServerArgs
from scratchpad.utils import (
    find_printable_text,
    get_exception_traceback,
    logger,
    kill_parent_process,
    get_zmq_socket,
)

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


@dataclasses.dataclass
class DecodeStatus:
    """Store the status of incremental decoding."""

    vid: int
    decoded_text: str
    decode_ids: List[int]
    surr_offset: int
    read_offset: int


class DetokenizerManager:
    """DetokenizerManager is a process that detokenizes the token ids."""

    def __init__(
        self,
        server_args: ServerArgs,
    ):
        # Init inter-process communication
        context = zmq.Context(2)
        self.recv_from_scheduler = get_zmq_socket(
            context, zmq.PULL, server_args.detokenizer_ipc_name
        )
        self.send_to_tokenizer = get_zmq_socket(
            context, zmq.PUSH, server_args.tokenizer_ipc_name
        )

        if server_args.skip_tokenizer_init:
            self.tokenizer = None
        else:
            self.tokenizer = get_tokenizer(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
            )

        self.decode_status = LimitedCapacityDict()

    def trim_matched_stop(
        self, output: Union[str, List[int]], finished_reason: Dict, no_stop_trim: bool
    ):
        if no_stop_trim or not finished_reason:
            return output

        matched = finished_reason.get("matched", None)
        if not matched:
            return output

        # TODO(lmzheng): handle the case where multiple stop strs are hit

        # Trim stop str.
        if isinstance(matched, str) and isinstance(output, str):
            pos = output.find(matched)
            return output[:pos] if pos != -1 else output

        # Trim stop token.
        if isinstance(matched, int) and isinstance(output, list):
            assert len(output) > 0
            return output[:-1]
        return output

    def event_loop(self):
        """The event loop that handles requests"""

        while True:
            recv_obj = self.recv_from_scheduler.recv_pyobj()

            if isinstance(recv_obj, BatchEmbeddingOut):
                # If it is embedding model, no detokenization is needed.
                self.send_to_tokenizer.send_pyobj(recv_obj)
                continue
            elif isinstance(recv_obj, UpdateWeightReqOutput):
                # If it is a weight update request, no detokenization is needed.
                self.send_to_tokenizer.send_pyobj(recv_obj)
                continue
            elif self.tokenizer is None:
                # If the tokenizer is skipped, no detokenization is needed
                self.send_to_tokenizer.send_pyobj(recv_obj)
                continue

            assert isinstance(recv_obj, BatchTokenIDOut)
            bs = len(recv_obj.rids)

            # Initialize decode status
            read_ids, surr_ids = [], []
            for i in range(bs):
                rid = recv_obj.rids[i]
                vid = recv_obj.vids[i]
                if rid not in self.decode_status or self.decode_status[rid].vid != vid:
                    s = DecodeStatus(
                        vid=vid,
                        decoded_text=recv_obj.decoded_texts[i],
                        decode_ids=recv_obj.decode_ids[i],
                        surr_offset=0,
                        read_offset=recv_obj.read_offsets[i],
                    )
                    self.decode_status[rid] = s
                else:
                    s = self.decode_status[rid]
                    s.decode_ids = recv_obj.decode_ids[i]

                read_ids.append(s.decode_ids[s.surr_offset :])
                surr_ids.append(s.decode_ids[s.surr_offset : s.read_offset])

            # TODO(lmzheng): handle skip_special_tokens/spaces_between_special_tokens per request
            surr_texts = self.tokenizer.batch_decode(
                surr_ids,
                skip_special_tokens=recv_obj.skip_special_tokens[0],
                spaces_between_special_tokens=recv_obj.spaces_between_special_tokens[0],
            )
            read_texts = self.tokenizer.batch_decode(
                read_ids,
                skip_special_tokens=recv_obj.skip_special_tokens[0],
                spaces_between_special_tokens=recv_obj.spaces_between_special_tokens[0],
            )

            # Incremental decoding
            output_strs = []
            for i in range(bs):
                s = self.decode_status[recv_obj.rids[i]]
                new_text = read_texts[i][len(surr_texts[i]) :]
                if recv_obj.finished_reasons[i] is None:
                    # Streaming chunk: update the decode status
                    if len(new_text) > 0 and not new_text.endswith("�"):
                        s.decoded_text = s.decoded_text + new_text
                        s.surr_offset = s.read_offset
                        s.read_offset = len(s.decode_ids)
                        new_text = ""
                    else:
                        new_text = find_printable_text(new_text)

                output_strs.append(
                    self.trim_matched_stop(
                        s.decoded_text + new_text,
                        recv_obj.finished_reasons[i],
                        recv_obj.no_stop_trim[i],
                    )
                )
            self.send_to_tokenizer.send_pyobj(
                BatchStrOut(
                    rids=recv_obj.rids,
                    finished_reasons=recv_obj.finished_reasons,
                    output_strs=output_strs,
                    prompt_tokens=recv_obj.prompt_tokens,
                    completion_tokens=recv_obj.completion_tokens,
                    cached_tokens=recv_obj.cached_tokens,
                    input_token_logprobs_val=recv_obj.input_token_logprobs_val,
                    input_token_logprobs_idx=recv_obj.input_token_logprobs_idx,
                    output_token_logprobs_val=recv_obj.output_token_logprobs_val,
                    output_token_logprobs_idx=recv_obj.output_token_logprobs_idx,
                    input_top_logprobs_val=recv_obj.input_top_logprobs_val,
                    input_top_logprobs_idx=recv_obj.input_top_logprobs_idx,
                    output_top_logprobs_val=recv_obj.output_top_logprobs_val,
                    output_top_logprobs_idx=recv_obj.output_top_logprobs_idx,
                    normalized_prompt_logprob=recv_obj.normalized_prompt_logprob,
                )
            )


class LimitedCapacityDict(OrderedDict):
    def __init__(self, capacity=1 << 15, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capacity = capacity

    def __setitem__(self, key, value):
        if len(self) >= self.capacity:
            # Remove the oldest element (first item in the dict)
            self.popitem(last=False)
        # Set the new item
        super().__setitem__(key, value)


def run_detokenizer_process(
    server_args: ServerArgs,
):
    try:
        manager = DetokenizerManager(server_args)
        manager.event_loop()
    except Exception:
        msg = get_exception_traceback()
        logger.error(msg)
        kill_parent_process()
