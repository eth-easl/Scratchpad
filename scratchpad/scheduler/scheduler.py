import os
import faulthandler
import zmq
import time
import torch
import warnings
import setproctitle
import threading
from concurrent import futures
import multiprocessing
from collections import deque
from dataclasses import asdict
from functools import cached_property
from typing import List, Optional, TYPE_CHECKING
from types import SimpleNamespace
from scratchpad.config.model_config import ModelConfig
from scratchpad.constrained.base_backend import create_grammar_backend
from scratchpad.nn.layers.logits_processor import LogitsProcessorOutput
from scratchpad.scheduler.schedule_batch import (
    FINISH_ABORT,
    BaseFinishReason,
    ImageInputs,
    Req,
    ScheduleBatch,
)
from scratchpad.scheduler.stats import Stats
from scratchpad.scheduler.policy_scheduler import (
    PrefillAdder,
    SchedulePolicy,
    AddReqResult,
)
from scratchpad.model_executor.speculative.spec_info import SpeculativeAlgorithm
from scratchpad.server.metrics import PrometheusStatLogger
from scratchpad.memory.chunk_cache import ChunkCache
from scratchpad.memory.radix_cache import RadixCache
from scratchpad.server.args import ServerArgs, global_args
from scratchpad.utils import (
    broadcast_pyobj,
    kill_parent_process,
    set_random_seed,
    get_processor,
    get_tokenizer,
    get_exception_traceback,
    logger,
    get_zmq_socket,
)
from ..managers.tp_worker import TpModelWorker
from ..managers.structs import (
    AbortReq,
    BatchEmbeddingOut,
    BatchTokenIDOut,
    FlushCacheReq,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UpdateWeightReqInput,
    UpdateWeightReqOutput,
    MemoryPoolControlReqInput,
    ProfileReq,
    GetMemPoolSizeReq,
    GetMemPoolSizeReqOutput,
    RegisterToppingsReqInput,
)

if TYPE_CHECKING:
    from scratchpad.server.metric_types import StatLoggerBase

# Crash on warning if we are running CI tests
crash_on_warning = os.getenv("SP_IS_IN_CI", "false") == "true"
test_retract = os.getenv("SP_TEST_RETRACT", "false") == "true"


class Scheduler:
    """A scheduler that manages a tensor parallel GPU worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        loggers: Optional[List["StatLoggerBase"]],
    ):
        # Parse args
        self.server_args = server_args
        self.tp_rank = tp_rank
        self.tp_size = server_args.tp_size
        self.schedule_policy = server_args.schedule_policy
        self.disable_jump_forward = server_args.disable_jump_forward
        self.spec_algorithm = SpeculativeAlgorithm.NONE
        self.max_toppings_per_batch = server_args.max_toppings_per_batch
        self.enable_overlap = server_args.enable_overlap_schedule
        self.skip_tokenizer_init = server_args.skip_tokenizer_init
        self.loggers = loggers
        self.decode_mem_cache_buf_multiplier = 1
        # update this ondemand
        # Init inter-process communication
        context = zmq.Context(2)

        if self.tp_rank == 0:
            self.recv_from_tokenizer = get_zmq_socket(
                context, zmq.PULL, server_args.scheduler_input_ipc_name
            )

            if server_args.skip_tokenizer_init:
                # Directly send to the tokenizer/api
                self.send_to_detokenizer = get_zmq_socket(
                    context, zmq.PUSH, server_args.tokenizer_ipc_name
                )
            else:
                # Send to the detokenizer
                self.send_to_detokenizer = get_zmq_socket(
                    context, zmq.PUSH, server_args.detokenizer_ipc_name
                )
        else:
            self.recv_from_tokenizer = None
            self.send_to_detokenizer = SimpleNamespace(send_pyobj=lambda x: None)

        # Init tokenizer
        self.model_config = ModelConfig(
            server_args.model_path,
            trust_remote_code=server_args.trust_remote_code,
            context_length=server_args.context_length,
            model_override_args=server_args.json_model_override_args,
            is_embedding=server_args.is_embedding,
        )
        self.is_generation = self.model_config.is_generation

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
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                )
        self.draft_worker = None
        # Launch a tensor parallel worker
        # if self.enable_overlap:
        #     TpWorkerClass = TpModelWorkerClient
        # else:
        #     TpWorkerClass = TpModelWorker

        self.tp_worker = TpModelWorker(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            nccl_port=server_args.nccl_ports[0],
        )

        # Get token and memory info from the model worker
        (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_req_len,
            self.max_req_input_len,
            self.random_seed,
            self.device,
            worker_global_server_args_dict,
            _,
            _,
            _,
        ) = self.tp_worker.get_worker_info()
        self.tp_cpu_group = self.tp_worker.get_tp_cpu_group()
        self.pad_input_ids_func = self.tp_worker.get_pad_input_ids_func()
        global_args.update(asdict(worker_global_server_args_dict))
        set_random_seed(self.random_seed)

        # Print debug info
        logger.info(
            f"max_total_num_tokens={self.max_total_num_tokens}, "
            f"max_prefill_tokens={self.max_prefill_tokens}, "
            f"max_running_requests={self.max_running_requests}, "
            f"context_len={self.model_config.context_len}"
        )

        # Init memory pool and cache
        self.req_to_token_pool, self.token_to_kv_pool = self.tp_worker.get_memory_pool()

        if (
            server_args.chunked_prefill_size is not None
            and server_args.disable_radix_cache
        ):
            self.tree_cache = ChunkCache(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool=self.token_to_kv_pool,
            )
        else:
            self.tree_cache = RadixCache(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool=self.token_to_kv_pool,
                disable=server_args.disable_radix_cache,
            )
        self.tree_cache_metrics = {"total": 0, "hit": 0}
        self.policy = SchedulePolicy(self.schedule_policy, self.tree_cache)

        # Init running status
        self.waiting_queue: List[Req] = []
        self.running_batch: Optional[ScheduleBatch] = None
        self.cur_batch: Optional[ScheduleBatch] = None
        self.last_batch: Optional[ScheduleBatch] = None

        self.forward_ct = 0
        self.forward_ct_decode = 0
        self.num_generated_tokens = 0
        self.last_stats_tic = time.time()
        self.stream_interval = server_args.stream_interval

        # Init chunked prefill
        self.chunked_prefill_size = server_args.chunked_prefill_size
        self.being_chunked_req = None
        self.is_mixed_chunk = (
            self.chunked_prefill_size is not None and server_args.enable_mixed_chunk
        )

        # Init the FSM cache for constrained generation
        self.grammar_queue: List[Req] = []
        if not server_args.skip_tokenizer_init:
            logger.info("Initializing grammar backend")
            self.grammar_backend = create_grammar_backend(
                server_args, self.tokenizer, self.model_config.vocab_size
            )
        else:
            self.grammar_backend = None

        # Init new token estimation
        assert (
            server_args.schedule_conservativeness >= 0
        ), "Invalid schedule_conservativeness"

        self.init_new_token_ratio = min(
            server_args.init_new_token_ratio * server_args.schedule_conservativeness,
            1.0,
        )
        self.min_new_token_ratio = min(
            self.init_new_token_ratio * server_args.base_min_new_token_ratio,
            1.0,
        )
        self.new_token_ratio_decay = (
            self.init_new_token_ratio - self.min_new_token_ratio
        ) / server_args.new_token_ratio_decay
        self.new_token_ratio = self.init_new_token_ratio

        self.batch_is_full = False

        # Init watchdog thread
        self.watchdog_timeout = server_args.watchdog_timeout
        t = threading.Thread(target=self.watchdog_thread, daemon=True)
        t.start()

        # Init profiler
        if os.getenv("SP_TORCH_PROFILER_DIR", "") == "":
            self.profiler = None
        else:
            self.torch_profiler_trace_dir = os.getenv("SP_TORCH_PROFILER_DIR")
            logger.info(
                "Profiling enabled. Traces will be saved to: %s",
                self.torch_profiler_trace_dir,
            )
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_stack=True,
            )
        logger.info(f"Scheduler initialized.")

    def watchdog_thread(self):
        self.watchdog_last_forward_ct = 0
        self.watchdog_last_time = time.time()

        while True:
            if self.cur_batch is not None:
                if self.watchdog_last_forward_ct == self.forward_ct:
                    if time.time() > self.watchdog_last_time + self.watchdog_timeout:
                        logger.error(f"Watchdog timeout ({self.watchdog_timeout=})")
                        break
                else:
                    self.watchdog_last_forward_ct = self.forward_ct
                    self.watchdog_last_time = time.time()
            time.sleep(self.watchdog_timeout / 2)

        kill_parent_process()

    @torch.inference_mode()
    def event_loop_normal(self):
        """A normal blocking scheduler loop."""
        self.last_batch = None

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)

            batch = self.get_next_batch_to_run()

            if self.server_args.enable_dp_attention:  # TODO: simplify this
                batch = self.prepare_dp_attn_batch(batch)

            self.cur_batch = batch

            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                # When the server is idle, so self-check and re-init some states
                self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio

            self.last_batch = batch

    @torch.inference_mode()
    def event_loop_overlap(self):
        """A scheduler loop that overlaps the CPU processing and GPU computation."""
        result_queue = deque()

        self.last_batch = None
        self.running_batch = None

        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch
            if batch:
                result = self.run_batch(batch)
                result_queue.append((batch.copy(), result))

            if self.last_batch:
                tmp_batch, tmp_result = result_queue.popleft()
                self.process_batch_result(tmp_batch, tmp_result)
            elif batch is None:
                self.check_memory()
                self.new_token_ratio = self.init_new_token_ratio

            self.last_batch = batch

    def recv_requests(self):
        if self.tp_rank == 0:
            recv_reqs = []

            while True:
                try:
                    recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break
                recv_reqs.append(recv_req)
        else:
            recv_reqs = None

        if self.tp_size != 1:
            recv_reqs = broadcast_pyobj(recv_reqs, self.tp_rank, self.tp_cpu_group)
        return recv_reqs

    def process_input_requests(self, recv_reqs: List):
        for recv_req in recv_reqs:
            if isinstance(recv_req, TokenizedGenerateReqInput):
                accepted = self.handle_generate_request(recv_req)
                if not accepted:
                    raise ValueError("Request rejected")
            elif isinstance(recv_req, TokenizedEmbeddingReqInput):
                self.handle_embedding_request(recv_req)
            elif isinstance(recv_req, FlushCacheReq):
                self.flush_cache()
            elif isinstance(recv_req, AbortReq):
                self.abort_request(recv_req)
            elif isinstance(recv_req, UpdateWeightReqInput):
                success, message = self.update_weights(recv_req)
                self.send_to_detokenizer.send_pyobj(
                    UpdateWeightReqOutput(success, message)
                )
            elif isinstance(recv_req, ProfileReq):
                if recv_req == ProfileReq.START_PROFILE:
                    self.start_profile()
                else:
                    self.stop_profile()
            elif isinstance(recv_req, GetMemPoolSizeReq):
                self.send_to_detokenizer.send_pyobj(
                    GetMemPoolSizeReqOutput(self.max_total_num_tokens)
                )
            elif isinstance(recv_req, MemoryPoolControlReqInput):
                self.tp_worker.expand_memory_pool(recv_req.delta)
            elif isinstance(recv_req, RegisterToppingsReqInput):
                self.tp_worker.register_toppings(recv_req)
            else:
                raise ValueError(f"Invalid request: {recv_req}")

    def handle_generate_request(
        self,
        recv_req: TokenizedGenerateReqInput,
    ):
        req = Req(
            recv_req.rid,
            recv_req.input_text,
            recv_req.input_ids,
            recv_req.sampling_params,
            topping_path=recv_req.topping_path,
        )
        req.tokenizer = self.tokenizer
        # Image inputs
        if recv_req.image_inputs is not None:
            req.image_inputs = ImageInputs.from_dict(
                recv_req.image_inputs, self.model_config.vocab_size
            )
            req.origin_input_ids = self.pad_input_ids_func(
                req.origin_input_ids_unpadded, req.image_inputs
            )

        req.return_logprob = recv_req.return_logprob
        req.top_logprobs_num = recv_req.top_logprobs_num
        req.stream = recv_req.stream
        req.logprob_start_len = recv_req.logprob_start_len

        if req.logprob_start_len == -1:
            # By default, only return the logprobs for output tokens
            req.logprob_start_len = len(recv_req.input_ids) - 1

        # Truncate prompts that are too long
        if len(req.origin_input_ids) > self.max_req_input_len:
            logger.error(
                "Request length is longer than the KV cache pool size or "
                "the max context length. Truncated!!!"
            )
            return False
            # req.origin_input_ids = req.origin_input_ids[: self.max_req_input_len]

        req.sampling_params.max_new_tokens = min(
            (
                req.sampling_params.max_new_tokens
                if req.sampling_params.max_new_tokens is not None
                else 1 << 30
            ),
            self.max_req_len - len(req.origin_input_ids) - 1,
        )

        # Init grammar cache for this request
        add_to_grammar_queue = False
        if (
            req.sampling_params.json_schema is not None
            or req.sampling_params.regex is not None
            or req.sampling_params.ebnf is not None
        ):
            assert self.grammar_backend is not None
            if req.sampling_params.json_schema is not None:
                key = ("json", req.sampling_params.json_schema)
            elif req.sampling_params.regex is not None:
                key = ("regex", req.sampling_params.regex)
            elif req.sampling_params.ebnf is not None:
                key = ("ebnf", req.sampling_params.ebnf)
            req.grammar = self.grammar_backend.get_cached_value(key)
            if not req.grammar:
                req.grammar = self.grammar_backend.get_future_value(key)
                add_to_grammar_queue = True

        if add_to_grammar_queue:
            self.grammar_queue.append(req)
        else:
            self.waiting_queue.append(req)
        return True

    def log_stats(self, stats):
        for logger in self.loggers:
            logger.log(stats)

    def handle_embedding_request(
        self,
        recv_req: TokenizedEmbeddingReqInput,
    ):
        req = Req(
            recv_req.rid,
            recv_req.input_text,
            recv_req.input_ids,
            recv_req.sampling_params,
        )
        req.tokenizer = self.tokenizer

        # Truncate prompts that are too long
        if len(req.origin_input_ids) >= self.max_req_input_len:
            logger.warning(
                "Request length is longer than the KV cache pool size or "
                "the max context length. Truncated!!!"
            )
            req.origin_input_ids = req.origin_input_ids[: self.max_req_input_len]

        self.waiting_queue.append(req)

    def log_decode_stats(self):
        num_used = self.max_total_num_tokens - (
            self.token_to_kv_pool.available_size() + self.tree_cache.evictable_size()
        )
        throughput = self.num_generated_tokens / (time.time() - self.last_stats_tic)
        stats = Stats(
            time.time(),
            generation_throughput=throughput,
            running_requests=len(self.running_batch.reqs),
            queued_requests=len(self.waiting_queue),
            token_usage=num_used / self.max_total_num_tokens,
            used_token_pool=num_used,
        )
        self.log_stats(stats)
        self.num_generated_tokens = 0
        self.last_stats_tic = time.time()
        num_running_reqs = len(self.running_batch.reqs) if self.running_batch else 0
        logger.info(
            f"Decode batch. "
            f"#running-req: {num_running_reqs}, "
            f"#token: {num_used}, "
            f"token usage: {num_used / self.max_total_num_tokens:.2f}, "
            f"gen throughput (token/s): {throughput:.2f}, "
            f"#queue-req: {len(self.waiting_queue)}"
        )

    def check_memory(self):
        available_size = (
            self.token_to_kv_pool.available_size() + self.tree_cache.evictable_size()
        )
        if available_size != self.max_total_num_tokens:
            warnings.warn(
                "Warning: "
                f"available_size={available_size}, max_total_num_tokens={self.max_total_num_tokens}\n"
                f"KV cache pool leak detected! pool size = {self.token_to_kv_pool.available_size()}, cache: {self.tree_cache.evictable_size()}"
            )
            exit(1) if crash_on_warning else None

        if len(self.req_to_token_pool.free_slots) != self.req_to_token_pool.size:
            warnings.warn(
                "Warning: "
                f"available req slots={len(self.req_to_token_pool.free_slots)}, "
                f"total slots={self.req_to_token_pool.size}\n"
                "Memory pool leak detected!"
            )
            exit(1) if crash_on_warning else None

    def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
        # Merge the prefill batch into the running batch
        if self.last_batch and self.last_batch.forward_mode.is_extend():
            if self.being_chunked_req:
                # Move the chunked request out of the batch
                self.last_batch.filter_batch(being_chunked_req=self.being_chunked_req)
                self.tree_cache.cache_unfinished_req(self.being_chunked_req)
                # being chunked request keeps its rid but will get a new req_pool_idx
                self.req_to_token_pool.free(self.being_chunked_req.req_pool_idx)
                self.batch_is_full = False

            if not self.last_batch.is_empty():
                if self.running_batch is None:
                    self.running_batch = self.last_batch
                else:
                    self.running_batch.merge_batch(self.last_batch)

        # Run prefill first if possible
        new_batch = self.get_new_batch_prefill()
        if new_batch is not None:
            return new_batch

        # Run decode
        if self.running_batch is None:
            return None
        self.running_batch = self.update_running_batch(self.running_batch)
        return self.running_batch

    def get_new_batch_prefill(self) -> Optional[ScheduleBatch]:
        if self.grammar_queue:
            self.move_ready_grammar_requests()
        # Handle the cases where prefill is not allowed
        if (
            self.batch_is_full or len(self.waiting_queue) == 0
        ) and self.being_chunked_req is None:
            return None

        running_bs = len(self.running_batch.reqs) if self.running_batch else 0
        if running_bs >= self.max_running_requests:
            self.batch_is_full = True
            return None

        # Get priority queue
        prefix_computed = self.policy.calc_priority(self.waiting_queue)

        # Prefill policy
        num_mixed_running = running_bs if self.is_mixed_chunk else 0
        adder = PrefillAdder(
            self.tree_cache,
            self.running_batch,
            self.new_token_ratio,
            self.token_to_kv_pool.available_size() + self.tree_cache.evictable_size(),
            self.max_prefill_tokens,
            self.chunked_prefill_size,
            num_mixed_running,
        )

        has_inflight = self.being_chunked_req is not None
        if has_inflight:
            self.being_chunked_req.init_next_round_input()
            self.being_chunked_req = adder.add_inflight_req(self.being_chunked_req)

        if self.topping_paths:
            topping_set = (
                set([req.topping_path for req in self.running_batch.reqs])
                if self.running_batch is not None
                else set([])
            )

        # Get requests from the waiting queue to a new prefill batch
        for req in self.waiting_queue:

            if (
                self.topping_paths
                and len(
                    topping_set
                    | set([req.topping_path for req in adder.can_run_list])
                    | set([req.topping_path])
                )
                > self.max_toppings_per_batch
            ):
                self.batch_is_full = True
                break

            if running_bs + len(adder.can_run_list) >= self.max_running_requests:
                self.batch_is_full = True
                break

            req.init_next_round_input(None if prefix_computed else self.tree_cache)
            res = adder.add_one_req(req)
            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    self.batch_is_full = True
                break

        # Update waiting queue
        can_run_list = adder.can_run_list
        if len(can_run_list) == 0:
            return None
        self.waiting_queue = [
            x for x in self.waiting_queue if x not in set(can_run_list)
        ]

        if adder.new_inflight_req is not None:
            assert self.being_chunked_req is None
            self.being_chunked_req = adder.new_inflight_req

        if self.being_chunked_req:
            self.being_chunked_req.is_being_chunked += 1

        # Print stats
        if self.tp_rank == 0:
            if isinstance(self.tree_cache, RadixCache):
                self.tree_cache_metrics["total"] += (
                    adder.log_input_tokens + adder.log_hit_tokens
                ) / 10**9
                self.tree_cache_metrics["hit"] += (adder.log_hit_tokens) / 10**9
                tree_cache_hit_rate = (
                    self.tree_cache_metrics["hit"] / self.tree_cache_metrics["total"]
                )
            else:
                tree_cache_hit_rate = 0.0

            num_used = self.max_total_num_tokens - (
                self.token_to_kv_pool.available_size()
                + self.tree_cache.evictable_size()
            )
            stats = Stats(
                now=time.time(),
                generation_throughput=0,
                token_usage=num_used / self.max_total_num_tokens,
                queued_requests=len(self.waiting_queue) + has_inflight,
                running_requests=num_mixed_running + running_bs,
                used_token_pool=num_used,
            )
            self.log_stats(stats)
            if num_mixed_running > 0:
                logger.info(
                    f"Prefill batch"
                    f"(mixed #running-req: {num_mixed_running}). "
                    f"#new-seq: {len(can_run_list)}, "
                    f"#new-token: {adder.log_input_tokens}, "
                    f"#cached-token: {adder.log_hit_tokens}, "
                    f"cache hit rate: {100.0 * tree_cache_hit_rate:.2f}%, "
                    f"token usage: {num_used / self.max_total_num_tokens:.2f}, "
                    f"#queue-req: {len(self.waiting_queue) + has_inflight}"
                )
            else:
                logger.info(
                    f"Prefill batch. "
                    f"#new-seq: {len(can_run_list)}, "
                    f"#new-token: {adder.log_input_tokens}, "
                    f"#cached-token: {adder.log_hit_tokens}, "
                    f"cache hit rate: {100.0 * tree_cache_hit_rate:.2f}%, "
                    f"token usage: {num_used / self.max_total_num_tokens:.2f}, "
                    f"#running-req: {running_bs}, "
                    f"#queue-req: {len(self.waiting_queue) + has_inflight}"
                )

        # Create a new batch
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            self.server_args.enable_custom_logit_processor,
        )
        new_batch.prepare_for_extend()

        # Mixed-style chunked prefill
        if self.is_mixed_chunk and self.running_batch is not None:
            self.running_batch.filter_batch()
            if not self.running_batch.is_empty():
                self.running_batch.prepare_for_decode(self.enable_overlap)
                new_batch.mix_with_running(self.running_batch)
                new_batch.decoding_reqs = self.running_batch.reqs
            self.running_batch = None
        else:
            new_batch.decoding_reqs = None
        return new_batch

    def update_running_batch(self, batch: ScheduleBatch) -> Optional[ScheduleBatch]:
        """Update the current running decoding batch."""
        global test_retract

        initial_bs = batch.batch_size()

        batch.filter_batch()
        if batch.is_empty():
            self.batch_is_full = False
            return None

        # Check if decode out of memory
        if not batch.check_decode_mem(self.decode_mem_cache_buf_multiplier) or (
            test_retract and batch.batch_size() > 10
        ):
            old_ratio = self.new_token_ratio

            retracted_reqs, new_token_ratio = batch.retract_decode()
            self.new_token_ratio = new_token_ratio
            if self.draft_worker:
                self.draft_worker.finish_request(retracted_reqs)

            logger.info(
                "Decode out of memory happened. "
                f"#retracted_reqs: {len(retracted_reqs)}, "
                f"#new_token_ratio: {old_ratio:.4f} -> {self.new_token_ratio:.4f}"
            )
            self.waiting_queue.extend(retracted_reqs)
        else:
            self.new_token_ratio = max(
                self.new_token_ratio - self.new_token_ratio_decay,
                self.min_new_token_ratio,
            )

        # Check for jump-forward
        if not self.disable_jump_forward:
            jump_forward_reqs = batch.check_for_jump_forward(self.pad_input_ids_func)
            self.waiting_queue.extend(jump_forward_reqs)
            if batch.is_empty():
                self.batch_is_full = False
                return None

        if batch.batch_size() < initial_bs:
            self.batch_is_full = False

        # Update batch tensors
        batch.prepare_for_decode(self, self.tp_worker.model_runner.topping_manager)
        return batch

    def run_batch(self, batch: ScheduleBatch):
        """Run a batch."""
        self.forward_ct += 1

        if self.is_generation:
            if batch.forward_mode.is_decode() or batch.extend_num_tokens != 0:
                if self.spec_algorithm.is_none():
                    model_worker_batch = batch.get_model_worker_batch()
                    (
                        logits_output,
                        next_token_ids,
                    ) = self.tp_worker.forward_batch_generation(model_worker_batch)
                else:
                    (
                        logits_output,
                        next_token_ids,
                        model_worker_batch,
                        num_accepted_tokens,
                    ) = self.draft_worker.forward_batch_speculative_generation(batch)
                    self.num_generated_tokens += num_accepted_tokens

            elif batch.forward_mode.is_idle():
                model_worker_batch = batch.get_model_worker_batch()
                self.tp_worker.forward_batch_idle(model_worker_batch)
                return
            else:
                logits_output = None
                if self.skip_tokenizer_init:
                    next_token_ids = torch.full(
                        (batch.batch_size(),), self.tokenizer.eos_token_id
                    )
                else:
                    next_token_ids = torch.full((batch.batch_size(),), 0)

            batch.output_ids = next_token_ids
            ret = logits_output, next_token_ids, model_worker_batch.bid
        else:  # embedding or reward model
            assert batch.extend_num_tokens != 0
            model_worker_batch = batch.get_model_worker_batch()
            embeddings = self.tp_worker.forward_batch_embedding(model_worker_batch)
            ret = embeddings, model_worker_batch.bid
        return ret

    def process_batch_result(self, batch: ScheduleBatch, result):
        if batch.forward_mode.is_decode():
            self.process_batch_result_decode(batch, result)
            if batch.is_empty():
                self.running_batch = None
        else:
            self.process_batch_result_prefill(batch, result)

    def process_batch_result_prefill(self, batch: ScheduleBatch, result):
        skip_stream_req = None

        if self.is_generation:
            logits_output, next_token_ids, bid = result

            if self.enable_overlap:
                logits_output, next_token_ids = self.tp_worker.resolve_batch_result(bid)
            else:
                # Move next_token_ids and logprobs to cpu
                next_token_ids = next_token_ids.tolist()
                if batch.return_logprob:
                    logits_output.next_token_logprobs = (
                        logits_output.next_token_logprobs.tolist()
                    )
                    logits_output.input_token_logprobs = (
                        logits_output.input_token_logprobs.tolist()
                    )
                    logits_output.normalized_prompt_logprobs = (
                        logits_output.normalized_prompt_logprobs.tolist()
                    )

            # Check finish conditions
            logprob_pt = 0
            for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
                if req.is_retracted:
                    continue

                if self.is_mixed_chunk and self.enable_overlap and req.finished():
                    # Free the one delayed token for the mixed decode batch
                    j = len(batch.out_cache_loc) - len(batch.reqs) + i
                    self.token_to_kv_pool.free(batch.out_cache_loc[j : j + 1])
                    continue

                if req.is_being_chunked <= 0:
                    req.output_ids.append(next_token_id)
                    req.check_finished()

                    if req.finished():
                        self.tree_cache.cache_finished_req(req)
                    elif not batch.decoding_reqs or req not in batch.decoding_reqs:
                        self.tree_cache.cache_unfinished_req(req)

                    if req.return_logprob:
                        logprob_pt += self.add_logprob_return_values(
                            i, req, logprob_pt, next_token_ids, logits_output
                        )

                    if req.grammar is not None:
                        req.grammar.accept_token(next_token_id)
                        req.grammar.finished = req.finished()
                else:
                    # being chunked reqs' prefill is not finished
                    req.is_being_chunked -= 1
                    # There is only at most one request being currently chunked.
                    # Because this request does not finish prefill,
                    # we don't want to stream the request currently being chunked.
                    skip_stream_req = req

            if batch.next_batch_sampling_info:
                batch.next_batch_sampling_info.update_regex_vocab_mask()
                self.current_stream.synchronize()
                batch.next_batch_sampling_info.sampling_info_done.set()

        else:  # embedding or reward model
            embeddings, bid = result
            embeddings = embeddings.tolist()

            # Check finish conditions
            for i, req in enumerate(batch.reqs):
                if req.is_retracted:
                    continue

                req.embedding = embeddings[i]
                if req.is_being_chunked <= 0:
                    # Dummy output token for embedding models
                    req.output_ids.append(0)
                    req.check_finished()

                    if req.finished():
                        self.tree_cache.cache_finished_req(req)
                    else:
                        self.tree_cache.cache_unfinished_req(req)
                else:
                    # being chunked reqs' prefill is not finished
                    req.is_being_chunked -= 1

        self.stream_output(batch.reqs, batch.return_logprob, skip_stream_req)

    def process_batch_result_decode(self, batch: ScheduleBatch, result):
        logits_output, next_token_ids, bid = result
        self.num_generated_tokens += len(batch.reqs)

        if self.enable_overlap:
            logits_output, next_token_ids = self.tp_worker.resolve_batch_result(bid)
            next_token_logprobs = logits_output.next_token_logprobs
        else:
            next_token_ids = next_token_ids.tolist()
            if batch.return_logprob:
                next_token_logprobs = logits_output.next_token_logprobs.tolist()

        self.token_to_kv_pool.free_group_begin()

        # Check finish condition
        for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
            if req.is_retracted:
                continue

            if self.enable_overlap and req.finished():
                # Free the one delayed token
                self.token_to_kv_pool.free(batch.out_cache_loc[i : i + 1])
                continue

            if batch.spec_algorithm.is_none():
                # speculative worker will solve the output_ids in speculative decoding
                req.output_ids.append(next_token_id)

            req.check_finished()

            if req.finished():
                self.tree_cache.cache_finished_req(req)

            if req.return_logprob:
                req.output_token_logprobs_val.append(next_token_logprobs[i])
                req.output_token_logprobs_idx.append(next_token_id)
                if req.top_logprobs_num > 0:
                    req.output_top_logprobs_val.append(
                        logits_output.next_token_top_logprobs_val[i]
                    )
                    req.output_top_logprobs_idx.append(
                        logits_output.next_token_top_logprobs_idx[i]
                    )

            if req.grammar is not None:
                req.grammar.accept_token(next_token_id)
                req.grammar.finished = req.finished()

        if batch.next_batch_sampling_info:
            batch.next_batch_sampling_info.update_regex_vocab_mask()
            self.current_stream.synchronize()
            batch.next_batch_sampling_info.sampling_info_done.set()

        self.stream_output(batch.reqs, batch.return_logprob)

        self.token_to_kv_pool.free_group_end()

        self.forward_ct_decode = (self.forward_ct_decode + 1) % (1 << 30)
        if (
            self.tp_rank == 0
            and self.forward_ct_decode % self.server_args.decode_log_interval == 0
        ):
            self.log_decode_stats()

    def add_logprob_return_values(
        self,
        i: int,
        req: Req,
        pt: int,
        next_token_ids: List[int],
        output: LogitsProcessorOutput,
    ):
        """Attach logprobs to the return values."""
        req.output_token_logprobs.append(
            (output.next_token_logprobs[i], next_token_ids[i])
        )

        # If logprob_start_len > 0, then first logprob_start_len prompt tokens will be ignored.
        num_input_logprobs = req.extend_input_len - req.extend_logprob_start_len

        if req.normalized_prompt_logprob is None:
            req.normalized_prompt_logprob = output.normalized_prompt_logprobs[i]

        if req.input_token_logprobs is None:
            input_token_logprobs = output.input_token_logprobs[
                pt : pt + num_input_logprobs - 1 - req.last_update_decode_tokens
            ]
            input_token_ids = req.fill_ids[
                len(req.fill_ids)
                - num_input_logprobs
                + 1 : len(req.fill_ids)
                - req.last_update_decode_tokens
            ]
            req.input_token_logprobs = list(zip(input_token_logprobs, input_token_ids))

            if (
                req.logprob_start_len == 0
            ):  # The first token does not have logprob, pad it.
                req.input_token_logprobs = [
                    (None, req.fill_ids[0])
                ] + req.input_token_logprobs

        if req.last_update_decode_tokens != 0:
            # Some decode tokens are re-computed in an extend batch
            req.output_token_logprobs.extend(
                list(
                    zip(
                        output.input_token_logprobs[
                            pt
                            + num_input_logprobs
                            - 1
                            - req.last_update_decode_tokens : pt
                            + num_input_logprobs
                            - 1
                        ],
                        req.fill_ids[
                            len(req.fill_ids)
                            - req.last_update_decode_tokens : len(req.fill_ids)
                        ],
                    )
                )
            )

        if req.top_logprobs_num > 0:
            if req.input_top_logprobs is None:
                req.input_top_logprobs = output.input_top_logprobs[i]
                if req.logprob_start_len == 0:
                    req.input_top_logprobs = [None] + req.input_top_logprobs

            if req.last_update_decode_tokens != 0:
                req.output_top_logprobs.extend(
                    output.input_top_logprobs[i][-req.last_update_decode_tokens :]
                )
            req.output_top_logprobs.append(output.output_top_logprobs[i])

        return num_input_logprobs

    def stream_output(
        self, reqs: List[Req], return_logprob: bool, skip_req: Optional[Req] = None
    ):
        """Stream the output to detokenizer."""
        rids = []
        finished_reasons: List[BaseFinishReason] = []

        if self.is_generation:
            vids = []
            decoded_texts = []
            decode_ids_list = []
            read_offsets = []
            output_ids = []

            skip_special_tokens = []
            spaces_between_special_tokens = []
            no_stop_trim = []
            prompt_tokens = []
            completion_tokens = []
            cached_tokens = []

            if return_logprob:
                input_token_logprobs_val = []
                input_token_logprobs_idx = []
                output_token_logprobs_val = []
                output_token_logprobs_idx = []
                input_top_logprobs_val = []
                input_top_logprobs_idx = []
                output_top_logprobs_val = []
                output_top_logprobs_idx = []
                normalized_prompt_logprob = []
            else:
                input_token_logprobs_val = (
                    input_token_logprobs_idx
                ) = (
                    output_token_logprobs_val
                ) = (
                    output_token_logprobs_idx
                ) = (
                    input_top_logprobs_val
                ) = (
                    input_top_logprobs_idx
                ) = (
                    output_top_logprobs_val
                ) = output_top_logprobs_idx = normalized_prompt_logprob = None

            for req in reqs:
                if req is skip_req:
                    continue

                # TODO(lianmin): revisit this for overlap + retract + stream
                if (
                    req.finished()
                    # If stream, follow the given stream_interval
                    or (req.stream and len(req.output_ids) % self.stream_interval == 0)
                    # If not stream, we still want to output some tokens to get the benefit of incremental decoding.
                    or (not req.stream and len(req.output_ids) % 50 == 0)
                ):
                    if self.draft_worker and req.finished():
                        self.draft_worker.finish_request(req)

                    rids.append(req.rid)
                    finished_reasons.append(
                        req.finished_reason.to_json() if req.finished_reason else None
                    )
                    vids.append(req.vid)
                    decoded_texts.append(req.decoded_text)
                    decode_ids, read_offset = req.init_incremental_detokenize()
                    decode_ids_list.append(decode_ids)
                    read_offsets.append(read_offset)
                    if self.skip_tokenizer_init:
                        output_ids.append(req.output_ids)
                    skip_special_tokens.append(req.sampling_params.skip_special_tokens)
                    spaces_between_special_tokens.append(
                        req.sampling_params.spaces_between_special_tokens
                    )
                    no_stop_trim.append(req.sampling_params.no_stop_trim)

                    prompt_tokens.append(len(req.origin_input_ids))
                    completion_tokens.append(len(req.output_ids))
                    cached_tokens.append(req.cached_tokens)

                    if return_logprob:
                        input_token_logprobs_val.append(req.input_token_logprobs_val)
                        input_token_logprobs_idx.append(req.input_token_logprobs_idx)
                        output_token_logprobs_val.append(req.output_token_logprobs_val)
                        output_token_logprobs_idx.append(req.output_token_logprobs_idx)
                        input_top_logprobs_val.append(req.input_top_logprobs_val)
                        input_top_logprobs_idx.append(req.input_top_logprobs_idx)
                        output_top_logprobs_val.append(req.output_top_logprobs_val)
                        output_top_logprobs_idx.append(req.output_top_logprobs_idx)
                        normalized_prompt_logprob.append(req.normalized_prompt_logprob)

            # Send to detokenizer
            if rids:
                self.send_to_detokenizer.send_pyobj(
                    BatchTokenIDOut(
                        rids,
                        finished_reasons,
                        vids,
                        decoded_texts,
                        decode_ids_list,
                        read_offsets,
                        output_ids,
                        skip_special_tokens,
                        spaces_between_special_tokens,
                        no_stop_trim,
                        prompt_tokens,
                        completion_tokens,
                        cached_tokens,
                        input_token_logprobs_val,
                        input_token_logprobs_idx,
                        output_token_logprobs_val,
                        output_token_logprobs_idx,
                        input_top_logprobs_val,
                        input_top_logprobs_idx,
                        output_top_logprobs_val,
                        output_top_logprobs_idx,
                        normalized_prompt_logprob,
                    )
                )
        else:  # embedding or reward model
            embeddings = []
            prompt_tokens = []
            for req in reqs:
                if req.finished():
                    rids.append(req.rid)
                    finished_reasons.append(req.finished_reason.to_json())
                    embeddings.append(req.embedding)
                    prompt_tokens.append(len(req.origin_input_ids))
            self.send_to_detokenizer.send_pyobj(
                BatchEmbeddingOut(rids, finished_reasons, embeddings, prompt_tokens)
            )

    def get_idle_batch(self):
        idle_batch = ScheduleBatch.init_new(
            [],
            self.req_to_token_pool,
            self.token_to_kv_pool,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
        )
        idle_batch.prepare_for_idle()
        return idle_batch

    def move_ready_grammar_requests(self):
        """Move requests whose grammar objects are ready from grammar_queue to waiting_queue."""
        num_ready_reqs = 0
        for req in self.grammar_queue:
            try:
                req.grammar = req.grammar.result(timeout=0.05)
                num_ready_reqs += 1
            except futures._base.TimeoutError:
                break

        if self.tp_size > 1:
            # Sync across TP ranks to make sure they have the same number of ready requests
            tensor = torch.tensor(num_ready_reqs, dtype=torch.int32)
            torch.distributed.all_reduce(
                tensor, op=torch.distributed.ReduceOp.MAX, group=self.tp_cpu_group
            )
            num_ready_reqs_max = tensor.item()
            for i in range(num_ready_reqs, num_ready_reqs_max):
                self.grammar_queue[i].grammar = self.grammar_queue[i].grammar.result()
            num_ready_reqs = num_ready_reqs_max

        self.waiting_queue.extend(self.grammar_queue[:num_ready_reqs])
        self.grammar_queue = self.grammar_queue[num_ready_reqs:]

    def flush_cache_wrapped(self, recv_req: FlushCacheReq):
        self.flush_cache()

    def flush_cache(self):
        """Flush the memory pool and cache."""
        if len(self.waiting_queue) == 0 and (
            self.running_batch is None or len(self.running_batch.reqs) == 0
        ):
            self.tree_cache.reset()
            self.tree_cache_metrics = {"total": 0, "hit": 0}
            if self.grammar_cache is not None:
                self.grammar_cache.reset()
            # TODO(dark): reset the bnf cache
            self.req_to_token_pool.clear()
            self.token_to_kv_pool.clear()
            torch.cuda.empty_cache()
            logger.info("Cache flushed successfully!")
            if_success = True
        else:
            logger.warning(
                f"Cache not flushed because there are pending requests. "
                f"#queue-req: {len(self.waiting_queue)}, "
                f"#running-req: {0 if self.running_batch is None else len(self.running_batch.reqs)}"
            )
            if_success = False
        return if_success

    def abort_request(self, recv_req: AbortReq):
        # Delete requests in the waiting queue
        to_del = None
        for i, req in enumerate(self.waiting_queue):
            if req.rid == recv_req.rid:
                to_del = i
                break

        if to_del is not None:
            del self.waiting_queue[to_del]

        # Delete requests in the running batch
        if self.running_batch:
            for req in self.running_batch.reqs:
                if req.rid == recv_req.rid and not req.finished():
                    req.finished_reason = FINISH_ABORT()
                    self.tree_cache.cache_finished_req(req)
                    break

    def update_weights(self, recv_req: UpdateWeightReqInput):
        """In-place update of the weights."""
        success, message = self.tp_worker.update_weights(recv_req)
        if success:
            flash_cache_success = self.flush_cache()
            assert flash_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        return success, message

    def start_profile(self) -> None:
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        self.profiler.start()

    def stop_profile(self) -> None:
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        self.profiler.stop()
        self.profiler.export_chrome_trace(
            self.torch_profiler_trace_dir + "/" + str(time.time()) + ".trace.json.gz"
        )
        logger.info("Profiler is done")

    @cached_property
    def topping_paths(self):
        return self.tp_worker.model_runner.topping_manager.toppings


def run_scheduler_process(
    server_args: ServerArgs,
    gpu_id: int,
    tp_rank: int,
    pipe_writer: multiprocessing.connection.Connection,
):
    try:
        setproctitle.setproctitle(f"sp:scheduler")
        faulthandler.enable()
        loggers = [PrometheusStatLogger(1, {"server_id": server_args.server_id}, 4096)]
        scheduler = Scheduler(
            server_args, gpu_id, tp_rank, dp_rank=None, loggers=loggers
        )
        pipe_writer.send("ready")
        scheduler.event_loop_normal()
    except ValueError as e:
        logger.info(f"Scheduler process exited: {e}")
    except Exception:
        msg = get_exception_traceback()
        logger.error(msg)
        kill_parent_process()
