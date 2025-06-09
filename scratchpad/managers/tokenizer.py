from typing import Dict, List, Optional, Tuple, Union, Any, Awaitable, TYPE_CHECKING
import dataclasses
import asyncio
import sys
import time
import signal
import zmq
import zmq.asyncio
import os
import copy
import fastapi
from fastapi import BackgroundTasks
import uvloop
from scratchpad.sampling.sampling_params import SamplingParams
from .structs import (
    AbortReq,
    BatchEmbeddingOut,
    BatchStrOut,
    BatchTokenIDOut,
    EmbeddingReqInput,
    FlushCacheReq,
    GenerateReqInput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UpdateWeightReqInput,
    UpdateWeightReqOutput,
    GetMemPoolSizeReq,
    GetMemPoolSizeReqOutput,
    ProfileReq,
    SessionParams,
    UpdateWeightFromDiskReqOutput,
)
from scratchpad.utils import (
    logger,
    get_tokenizer,
    get_processor,
    get_zmq_socket,
    kill_child_process,
    dataclass_to_string_truncated,
    RWLock,
)
from scratchpad.config.model_config import ModelConfig
from .image_processor import get_dummy_image_processor, get_image_processor

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
if TYPE_CHECKING:
    from scratchpad.server.args import ServerArgs


@dataclasses.dataclass
class ReqState:
    """Store the state a request."""

    out_list: List[Dict[Any, Any]]
    finished: bool
    event: asyncio.Event
    obj: Union[GenerateReqInput, EmbeddingReqInput]

    # For metrics
    created_time: float
    finished_time: float = 0.0
    first_token_time: float = 0.0
    last_time: float = 0.0
    last_completion_tokens: int = 1

    # For streaming output
    last_output_offset: int = 0
    # For incremental state update.
    text: str = ""
    output_ids: List[int] = dataclasses.field(default_factory=list)
    input_token_logprobs_val: List[float] = dataclasses.field(default_factory=list)
    input_token_logprobs_idx: List[int] = dataclasses.field(default_factory=list)
    output_token_logprobs_val: List[float] = dataclasses.field(default_factory=list)
    output_token_logprobs_idx: List[int] = dataclasses.field(default_factory=list)
    input_top_logprobs_val: List[List[float]] = dataclasses.field(default_factory=list)
    input_top_logprobs_idx: List[List[int]] = dataclasses.field(default_factory=list)
    output_top_logprobs_val: List[List[float]] = dataclasses.field(default_factory=list)
    output_top_logprobs_idx: List[List[int]] = dataclasses.field(default_factory=list)
    input_token_ids_logprobs_val: List = dataclasses.field(default_factory=list)
    input_token_ids_logprobs_idx: List = dataclasses.field(default_factory=list)
    output_token_ids_logprobs_val: List = dataclasses.field(default_factory=list)
    output_token_ids_logprobs_idx: List = dataclasses.field(default_factory=list)


class TokenizerManager:
    """TokenizerManager is a process that tokenizes the text."""

    def __init__(
        self,
        server_args: "ServerArgs",
    ):
        # Parse args
        self.server_args = server_args

        # Init inter-process communication
        context = zmq.asyncio.Context(2)
        self.recv_from_detokenizer = get_zmq_socket(
            context,
            zmq.PULL,
            server_args.tokenizer_ipc_name,
            True,
        )
        self.send_to_scheduler = get_zmq_socket(
            context,
            zmq.PUSH,
            server_args.scheduler_input_ipc_name,
            True,
        )

        # Read model args
        self.model_path = server_args.model_path
        self.served_model_name = server_args.served_model_name
        self.model_config = ModelConfig(
            server_args.model_path,
            trust_remote_code=server_args.trust_remote_code,
            context_length=server_args.context_length,
            model_override_args=server_args.json_model_override_args,
            is_embedding=server_args.is_embedding,
        )

        self.is_generation = self.model_config.is_generation
        self.context_len = self.model_config.context_len

        # Create image processor placeholder
        self.image_processor = get_dummy_image_processor()

        # Create tokenizer
        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if self.model_config.is_multimodal:
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                )
                self.tokenizer = self.processor.tokenizer
                os.environ["TOKENIZERS_PARALLELISM"] = "false"

                # We want to parallelize the image pre-processing so we create an executor for it
                self.image_processor = get_image_processor(
                    self.model_config.hf_config, server_args, self.processor
                )
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                )

        # Store states
        self.to_create_loop = True
        self.rid_to_state: Dict[str, ReqState] = {}

        self.model_update_lock = RWLock()
        self.model_update_result: Optional[
            Awaitable[UpdateWeightFromDiskReqOutput]
        ] = None
        self.asyncio_tasks = set()

        # Others
        self.gracefully_exit = False

    async def generate_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        created_time = time.time()

        self.auto_create_handle_loop()

        if isinstance(obj, EmbeddingReqInput) and self.is_generation:
            raise ValueError(
                "This model does not appear to be an embedding model by default. "
                "Please add `--is-embedding` when launching the server or try another model."
            )
        obj.normalize_batch_and_arguments()

        if self.server_args.log_requests:
            logger.info(f"Receive: obj={dataclass_to_string_truncated(obj)}")

        async with self.model_update_lock.reader_lock:
            is_single = obj.is_single
            if is_single:
                tokenized_obj = await self._tokenize_one_request(obj)
                self._send_one_request(obj, tokenized_obj, created_time)
                async for response in self._wait_one_response(obj, request):
                    yield response
            else:
                async for response in self._handle_batch_request(
                    obj, request, created_time
                ):
                    yield response

    async def _tokenize_one_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
    ):
        """Tokenize one request."""
        # Tokenize
        input_embeds = None
        input_text = obj.text
        if obj.input_embeds is not None:
            if not self.server_args.disable_radix_cache:
                raise ValueError(
                    "input_embeds is provided while disable_radix_cache is False. "
                    "Please add `--disable-radix-cache` when you launch the server "
                    "if you want to use input_embeds as inputs."
                )
            input_embeds = obj.input_embeds
            input_ids = obj.input_ids
        elif obj.input_ids is None:
            input_ids = self.tokenizer.encode(input_text)
        else:
            input_ids = obj.input_ids

        if self.is_generation:
            # TODO: also support getting embeddings for multimodal models
            image_inputs: Dict = await self.image_processor.process_images_async(
                obj.image_data, input_text or input_ids, obj
            )
            if image_inputs and "input_ids" in image_inputs:
                input_ids = image_inputs["input_ids"]
            return_logprob = obj.return_logprob
            logprob_start_len = obj.logprob_start_len
            top_logprobs_num = obj.top_logprobs_num
            token_ids_logprob = obj.token_ids_logprob
            session_params = (
                SessionParams(**obj.session_params) if obj.session_params else None
            )

        if obj.input_ids is not None and len(input_ids) >= self.context_len:
            raise ValueError(
                f"The input ({len(input_ids)} tokens) is longer than the "
                f"model's context length ({self.context_len} tokens)."
            )

        # Parse sampling parameters
        sampling_params = SamplingParams(**obj.sampling_params)
        sampling_params.normalize(self.tokenizer)
        sampling_params.verify()

        # Build return object
        if isinstance(obj, GenerateReqInput):
            tokenized_obj = TokenizedGenerateReqInput(
                obj.rid,
                input_text,
                input_ids,
                image_inputs,
                sampling_params,
                return_logprob,
                logprob_start_len,
                top_logprobs_num,
                token_ids_logprob,
                obj.stream,
                topping_path=obj.topping_path,
                input_embeds=input_embeds,
                session_params=session_params,
                custom_logit_processor=obj.custom_logit_processor,
                return_hidden_states=obj.return_hidden_states,
            )
        elif isinstance(obj, EmbeddingReqInput):
            tokenized_obj = TokenizedEmbeddingReqInput(
                obj.rid,
                input_text,
                input_ids,
                sampling_params,
            )

        return tokenized_obj

    def _send_one_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        tokenized_obj: Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput],
        created_time: Optional[float] = None,
    ):
        event = asyncio.Event()
        state = ReqState([], False, event, obj, created_time=created_time)
        self.rid_to_state[obj.rid] = state
        self.send_to_scheduler.send_pyobj(tokenized_obj)

    async def _wait_one_response(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        """Wait for the response of one request."""
        state = self.rid_to_state[obj.rid]

        while True:
            try:
                await asyncio.wait_for(state.event.wait(), timeout=4)
            except asyncio.TimeoutError:
                if request is not None and await request.is_disconnected():
                    self.abort_request(obj.rid)
                    raise ValueError(f"Abort request {obj.rid}")
                continue

            out = state.out_list[-1]

            state.out_list = []
            if state.finished:
                if self.server_args.log_requests:
                    msg = f"Finish: obj={dataclass_to_string_truncated(obj)}, out={dataclass_to_string_truncated(out)}"
                    logger.info(msg)
                del self.rid_to_state[obj.rid]
                yield out
                break

            state.event.clear()

            if obj.stream:
                yield out
            else:
                if request is not None and await request.is_disconnected():
                    self.abort_request(obj.rid)
                    raise ValueError(f"Abort request {obj.rid}")

    async def _handle_batch_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
        created_time: Optional[float] = None,
    ):
        batch_size = obj.batch_size

        generators = []
        rids = []
        if getattr(obj, "parallel_sample_num", 1) == 1:
            # Send all requests
            for i in range(batch_size):
                tmp_obj = obj[i]
                tokenized_obj = await self._tokenize_one_request(tmp_obj)
                self._send_one_request(tmp_obj, tokenized_obj, created_time)
                generators.append(self._wait_one_response(tmp_obj, request))
                rids.append(tmp_obj.rid)
        else:
            # FIXME: When using batch and parallel_sample_num together, the perf is not optimal.
            if batch_size > 128:
                logger.warning(
                    "Sending a single large batch with parallel sampling (n > 1) has not been well optimized. "
                    "The performance might be better if you just duplicate the requests n times or use "
                    "many threads to send them one by one with parallel sampling (n > 1)."
                )

            # Tokenize all requests
            objs = [obj[i] for i in range(batch_size)]
            tokenized_objs = await asyncio.gather(
                *(self._tokenize_one_request(obj) for obj in objs)
            )

            # Cache the common prefix for parallel sampling
            for i in range(batch_size):
                tmp_obj = copy.copy(objs[i])
                tokenized_obj = copy.copy(tokenized_objs[i])
                tokenized_obj.rid = tmp_obj.regenerate_rid()
                tokenized_obj.sampling_params = copy.copy(tokenized_obj.sampling_params)
                tokenized_obj.sampling_params.max_new_tokens = 0
                tokenized_obj.stream = False
                self._send_one_request(tmp_obj, tokenized_obj, created_time)
                await self._wait_one_response(tmp_obj, request).__anext__()

            # Expand requests, assign new rids for them, and send them
            for i in range(batch_size):
                for _ in range(obj.parallel_sample_num):
                    tmp_obj = copy.copy(objs[i])
                    tokenized_obj = copy.copy(tokenized_objs[i])
                    tokenized_obj.rid = tmp_obj.regenerate_rid()
                    self._send_one_request(tmp_obj, tokenized_obj, created_time)
                    generators.append(self._wait_one_response(tmp_obj, request))
                    rids.append(tmp_obj.rid)

        # Wait for all requests
        is_stream = hasattr(obj, "stream") and obj.stream
        if not is_stream:
            outputs = await asyncio.gather(*(gen.__anext__() for gen in generators))
            yield outputs
        else:
            rid_to_index = {rid: i for i, rid in enumerate(rids)}
            task_map = {asyncio.create_task(gen.__anext__()): gen for gen in generators}
            while task_map:
                done, _ = await asyncio.wait(
                    task_map.keys(), return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    gen = task_map.pop(task)
                    try:
                        result = task.result()
                        result["index"] = rid_to_index[result["meta_info"]["id"]]
                        yield result
                        new_task = asyncio.create_task(gen.__anext__())
                        task_map[new_task] = gen
                    except StopAsyncIteration:
                        pass

    def flush_cache(self):
        req = FlushCacheReq()
        self.send_to_scheduler.send_pyobj(req)

    def abort_request(self, rid: str):
        if rid not in self.rid_to_state:
            return
        del self.rid_to_state[rid]
        req = AbortReq(rid)
        self.send_to_scheduler.send_pyobj(req)

    def start_profile(self):
        req = ProfileReq.START_PROFILE
        self.send_to_scheduler.send_pyobj(req)

    def stop_profile(self):
        req = ProfileReq.STOP_PROFILE
        self.send_to_scheduler.send_pyobj(req)

    async def get_memory_pool_size(self):
        if self.to_create_loop:
            self.create_handle_loop()

        req = GetMemPoolSizeReq()

        self.send_to_scheduler.send_pyobj(req)
        self.mem_pool_size = asyncio.Future()

        # FIXME: Each request should have its own future instead of using `self.mem_pool_size`.
        if self.server_args.dp_size == 1:
            res = await self.mem_pool_size
            return res.size
        else:  # self.server_args.dp_size > 1
            self.mem_pool_size_tmp = []
            res = await self.mem_pool_size
            ret = [r.size for r in res]
            return ret

    async def update_weights(
        self, obj: UpdateWeightReqInput, request: Optional[fastapi.Request] = None
    ):
        if self.to_create_loop:
            self.create_handle_loop()

        # default the load format to the server_args
        if obj.load_format is None:
            obj.load_format = self.server_args.load_format

        if not self.model_update_lock.locked():

            async with self.model_update_lock:
                # wait for the previous generation requests to finish
                while len(self.rid_to_state) > 0:
                    await asyncio.sleep(0.001)
                self.send_to_scheduler.send_pyobj(obj)
                self.model_update_result = asyncio.Future()

                if self.server_args.dp_size == 1:
                    result = await self.model_update_result
                    if result.success:
                        self.server_args.model_path = obj.model_path
                        self.server_args.load_format = obj.load_format
                        self.model_path = obj.model_path
                    return result.success, result.message
                else:  # self.server_args.dp_size > 1
                    self.model_update_tmp = []
                    result = await self.model_update_result

                    all_success = all([r.success for r in result])
                    if all_success is True:
                        self.server_args.model_path = obj.model_path
                        self.server_args.load_format = obj.load_format
                        self.model_path = obj.model_path
                    all_message = [r.message for r in result]
                    all_message = " | ".join(all_message)
                    return all_success, all_message

        else:
            return False, "Another update is in progress. Please try again later."

    def create_abort_task(self, obj: GenerateReqInput):
        # Abort the request if the client is disconnected.
        async def abort_request():
            await asyncio.sleep(1)
            if obj.is_single:
                self.abort_request(obj.rid)
            else:
                for rid in obj.rid:
                    self.abort_request(rid)

        background_tasks = BackgroundTasks()
        background_tasks.add_task(abort_request)
        return background_tasks

    def auto_create_handle_loop(self):
        if not self.to_create_loop:
            return

        self.to_create_loop = False
        loop = asyncio.get_event_loop()
        self.asyncio_tasks.add(loop.create_task(self.handle_loop()))

        signal_handler = SignalHandler(self)
        loop.add_signal_handler(signal.SIGTERM, signal_handler.signal_handler)
        self.asyncio_tasks.add(loop.create_task(self.sigterm_watchdog()))

    async def sigterm_watchdog(self):
        while not self.gracefully_exit:
            await asyncio.sleep(5)

        # drain requests
        while True:
            remain_num_req = len(self.rid_to_state)
            logger.info(
                f"Gracefully exiting... remaining number of requests {remain_num_req}"
            )
            if remain_num_req > 0:
                await asyncio.sleep(5)
            else:
                break

        kill_child_process(include_self=True)
        sys.exit(-1)

    async def handle_loop(self):
        """The event loop that handles requests"""

        while True:
            recv_obj: Union[
                BatchStrOut, BatchEmbeddingOut, BatchTokenIDOut, UpdateWeightReqOutput
            ] = await self.recv_from_detokenizer.recv_pyobj()

            if isinstance(recv_obj, UpdateWeightReqOutput):
                if self.server_args.dp_size == 1:
                    self.model_update_result.set_result(recv_obj)
                else:  # self.server_args.dp_size > 1
                    self.model_update_tmp.append(recv_obj)
                    # set future if the all results are recevied
                    if len(self.model_update_tmp) == self.server_args.dp_size:
                        self.model_update_result.set_result(self.model_update_tmp)
                continue

            elif isinstance(recv_obj, GetMemPoolSizeReqOutput):
                if self.server_args.dp_size == 1:
                    self.mem_pool_size.set_result(recv_obj)
                else:  # self.sever_args.dp_size > 1
                    self.mem_pool_size_tmp.append(recv_obj)
                    # set future if the all results are received
                    if len(self.mem_pool_size_tmp) == self.server_args.dp_size:
                        self.mem_pool_size.set_result(self.mem_pool_size_tmp)
                continue

            assert isinstance(
                recv_obj, (BatchStrOut, BatchEmbeddingOut, BatchTokenIDOut)
            ), f"Unexpected obj received: {type(recv_obj)}"

            for i, rid in enumerate(recv_obj.rids):
                state = self.rid_to_state.get(rid, None)
                if state is None:
                    continue
                meta_info = {
                    "id": rid,
                    "finish_reason": recv_obj.finished_reasons[i],
                    "prompt_tokens": recv_obj.prompt_tokens[i],
                }
                if getattr(state.obj, "return_logprob", False):
                    self.convert_logprob_style(
                        meta_info,
                        state,
                        state.obj.top_logprobs_num,
                        state.obj.token_ids_logprob,
                        state.obj.return_text_in_logprobs
                        and not self.server_args.skip_tokenizer_init,
                        recv_obj,
                        i,
                    )
                if not isinstance(recv_obj, BatchEmbeddingOut):
                    meta_info.update(
                        {
                            "completion_tokens": recv_obj.completion_tokens[i],
                            "cached_tokens": recv_obj.cached_tokens[i],
                        }
                    )
                if isinstance(recv_obj, BatchStrOut):
                    out_dict = {
                        "text": recv_obj.output_strs[i],
                        "meta_info": meta_info,
                    }
                elif isinstance(recv_obj, BatchTokenIDOut):
                    out_dict = {
                        "token_ids": recv_obj.output_ids[i],
                        "meta_info": meta_info,
                    }
                else:
                    assert isinstance(recv_obj, BatchEmbeddingOut)

                    out_dict = {
                        "embedding": recv_obj.embeddings[i],
                        "meta_info": meta_info,
                    }

                state.out_list.append(out_dict)
                state.finished = recv_obj.finished_reasons[i] is not None
                state.event.set()

    def convert_logprob_style(
        self,
        meta_info: dict,
        state: ReqState,
        top_logprobs_num: int,
        token_ids_logprob: List[int],
        return_text_in_logprobs: bool,
        recv_obj: BatchStrOut,
        recv_obj_index: int,
    ) -> None:
        if len(recv_obj.input_token_logprobs_val) > 0:
            state.input_token_logprobs_val.extend(
                recv_obj.input_token_logprobs_val[recv_obj_index]
            )
            state.input_token_logprobs_idx.extend(
                recv_obj.input_token_logprobs_idx[recv_obj_index]
            )
        state.output_token_logprobs_val.extend(
            recv_obj.output_token_logprobs_val[recv_obj_index]
        )
        state.output_token_logprobs_idx.extend(
            recv_obj.output_token_logprobs_idx[recv_obj_index]
        )
        meta_info["input_token_logprobs"] = self.detokenize_logprob_tokens(
            state.input_token_logprobs_val,
            state.input_token_logprobs_idx,
            return_text_in_logprobs,
        )
        meta_info["output_token_logprobs"] = self.detokenize_logprob_tokens(
            state.output_token_logprobs_val,
            state.output_token_logprobs_idx,
            return_text_in_logprobs,
        )

        if top_logprobs_num > 0:
            if len(recv_obj.input_top_logprobs_val) > 0:
                state.input_top_logprobs_val.extend(
                    recv_obj.input_top_logprobs_val[recv_obj_index]
                )
                state.input_top_logprobs_idx.extend(
                    recv_obj.input_top_logprobs_idx[recv_obj_index]
                )
            state.output_top_logprobs_val.extend(
                recv_obj.output_top_logprobs_val[recv_obj_index]
            )
            state.output_top_logprobs_idx.extend(
                recv_obj.output_top_logprobs_idx[recv_obj_index]
            )
            meta_info["input_top_logprobs"] = self.detokenize_top_logprobs_tokens(
                state.input_top_logprobs_val,
                state.input_top_logprobs_idx,
                return_text_in_logprobs,
            )
            meta_info["output_top_logprobs"] = self.detokenize_top_logprobs_tokens(
                state.output_top_logprobs_val,
                state.output_top_logprobs_idx,
                return_text_in_logprobs,
            )

        if token_ids_logprob is not None:
            if len(recv_obj.input_token_ids_logprobs_val) > 0:
                state.input_token_ids_logprobs_val.extend(
                    recv_obj.input_token_ids_logprobs_val[recv_obj_index]
                )
                state.input_token_ids_logprobs_idx.extend(
                    recv_obj.input_token_ids_logprobs_idx[recv_obj_index]
                )
            state.output_token_ids_logprobs_val.extend(
                recv_obj.output_token_ids_logprobs_val[recv_obj_index]
            )
            state.output_token_ids_logprobs_idx.extend(
                recv_obj.output_token_ids_logprobs_idx[recv_obj_index]
            )
            meta_info["input_token_ids_logprobs"] = self.detokenize_top_logprobs_tokens(
                state.input_token_ids_logprobs_val,
                state.input_token_ids_logprobs_idx,
                return_text_in_logprobs,
            )
            meta_info[
                "output_token_ids_logprobs"
            ] = self.detokenize_top_logprobs_tokens(
                state.output_token_ids_logprobs_val,
                state.output_token_ids_logprobs_idx,
                return_text_in_logprobs,
            )

    def detokenize_logprob_tokens(
        self,
        token_logprobs_val: List[float],
        token_logprobs_idx: List[int],
        decode_to_text: bool,
    ):
        if not decode_to_text:
            return [
                (logprob, token_id, None)
                for logprob, token_id in zip(token_logprobs_val, token_logprobs_idx)
            ]
        else:
            assert self.tokenizer is not None
            token_texts = self.tokenizer.batch_decode(token_logprobs_idx)
            return list(zip(token_logprobs_val, token_logprobs_idx, token_texts))

    def detokenize_top_logprobs_tokens(
        self,
        token_logprobs_val: List[float],
        token_logprobs_idx: List[int],
        decode_to_text: bool,
    ):
        # TODO: The current implementation only batches the detokenization for top-k tokens per single position.
        # We should batch all top-k tokens in all positions.
        ret = []
        for i in range(len(token_logprobs_val)):
            if token_logprobs_val[i]:
                ret.append(
                    self.detokenize_logprob_tokens(
                        token_logprobs_val[i], token_logprobs_idx[i], decode_to_text
                    )
                )
            else:
                ret.append(None)
        return ret


class SignalHandler:
    def __init__(self, tokenizer_manager):
        self.tokenizer_manager = tokenizer_manager

    def signal_handler(self, signum=None, frame=None):
        logger.warning(
            f"SIGTERM received. {signum=} {frame=}. Draining requests and shutting down..."
        )
        self.tokenizer_manager.gracefully_exit = True
