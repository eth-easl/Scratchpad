from copy import deepcopy
from collections import deque
from internal.analyzer import ModelAnalyzer
from .trace import TraceEvent
from .memory_planner import MemoryPlanner
from internal.configs.hardware_params import hardware_params
from typing import List, Deque

from .request import GenerationRequest


class LLMEngine:
    def __init__(self, engine_id, model_name, hardware_name, w_bit, a_bit, kv_bit):
        self.engine_id = engine_id
        self.model_name = model_name
        self.hardware_name = hardware_name
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.kv_bit = kv_bit
        self.analyzer = ModelAnalyzer(
            model_id=model_name,
            hardware=hardware_name,
            config_file="adaml/internal/configs/llama.py",
            source="huggingface",
        )
        self.waiting: Deque[GenerationRequest] = deque()
        self.running: Deque[GenerationRequest] = deque()
        self.finished: List[GenerationRequest] = []
        self.failed: List[GenerationRequest] = []
        self.memory_planner = MemoryPlanner(
            self.analyzer.model_params,
            hardware_params[hardware_name],
            w_bit,
            a_bit,
            kv_bit,
        )
        self.memory_planner.print_status()
        self.finished_requests: int = 0
        self.configure()

    def configure(self):
        pass

    def add_request(self, request: GenerationRequest):
        self.waiting.append(request)

    def _prefill(self, request: GenerationRequest, start_at: float):
        self.memory_planner.allocate(request)
        memory_event = self.memory_event(start_at)
        if start_at < request.arrive_at:
            start_at = request.arrive_at
        self.running.append(request)
        request._prefill()
        prefill_result = self.analyzer.analyze(
            seqlen=request.input_length,
            batchsize=1,
            w_bit=self.w_bit,
            a_bit=self.a_bit,
            kv_bit=self.kv_bit,
        )
        prefill_time = prefill_result["total_results"]["prefill"]["inference_time"]
        request.set_prefill_finished_at(start_at + prefill_time)
        if request.output_length == 1:
            request.set_generation_finished_at(start_at + prefill_time)
            self.memory_planner.free([request.req_id])
        return prefill_time + start_at, [request], memory_event

    def _decode(self, requests: List[GenerationRequest], start_at: float):
        max_batch_size = len(requests)
        decode_time = []
        finished_requests_in_this_batch = []
        executable_requests = []
        for req in requests:
            if self.memory_planner.can_allocate_request(req):
                self.memory_planner.allocate(req)
                executable_requests.append(req)
        batch_size = len(executable_requests)
        memory_event = self.memory_event(start_at)
        for req in executable_requests:
            if start_at < req.arrive_at:
                start_at = req.arrive_at
            decode_result = self.analyzer.analyze(
                req.input_length + req.generated_tokens,
                batchsize=max_batch_size,
                w_bit=self.w_bit,
                a_bit=self.a_bit,
                kv_bit=self.kv_bit,
            )
            decode_time.append(
                decode_result["total_results"]["decode"]["inference_time"]
            )
        finished_at = max(decode_time) + start_at
        for req in executable_requests:
            finished = req._decode()
            if finished:
                req.set_generation_finished_at(finished_at)
                self.finished_requests += 1
                self.running.remove(req)
                self.finished.append(req)
                finished_requests_in_this_batch.append(req.req_id)
        self.memory_planner.free(finished_requests_in_this_batch)
        return finished_at, executable_requests, memory_event

    def step(self, start_at: float):
        # let's assume that process one request per step is fine in terms of utilization
        handled_requests = []
        # self.memory_planner.print_status()

        if len(self.waiting) > 0:
            # TODO(xiaozhe): this logic does not handle the case where
            # a single input is too long to fit in the memory
            if self.memory_planner.can_allocate_request(self.waiting[0]):
                pending_req = self.waiting.popleft()
                handled_requests = [pending_req.req_id]
                prefill_end_at, handled_requests, memory_event = self._prefill(
                    pending_req, start_at
                )
                return (
                    self.create_event(
                        "prefill", handled_requests, start_at, prefill_end_at
                    ),
                    prefill_end_at,
                    memory_event,
                )
            else:
                self.failed.append(self.waiting.popleft())
                return None, start_at + 0.0001, None

        elif len(self.running) > 0:
            # if there's no request needs prefill, proceed to decode
            # TODO(xiaozhe): let's assume we could do infinite batch size...
            decode_finished_at, handled_requests, memory_event = self._decode(
                list(self.running), start_at
            )
            return (
                self.create_event(
                    "decode", handled_requests, start_at, decode_finished_at
                ),
                decode_finished_at,
                memory_event,
            )
        else:
            # add a shift to the timer,
            # since we need to move on
            return None, start_at + 0.0001, None

    def create_event(self, phase, handled_requests, start_at, end_at):
        complete_events = []
        handled_requests = [req.req_id for req in handled_requests]
        for req in handled_requests:
            complete = TraceEvent(
                name=f"{phase}-{req}",
                cat=f"{phase,req}",
                ph="X",
                pid=self.engine_id,
                tid=0,
                ts=int(start_at * 1000 * 1000),  # convert to microseconds
                dur=int((end_at - start_at) * 1000 * 1000),
            )
            complete_events.append(complete)
        return complete_events

    def memory_event(self, start_at):
        return TraceEvent(
            name="block usage",
            ph="C",
            ts=start_at * 1e6,
            pid=self.engine_id,
            tid=0,
            cat="memory",
            args={
                "used": self.memory_planner._allocated_blocks,
                "free": self.memory_planner._max_num_blocks
                - self.memory_planner._allocated_blocks,
            },
        )

    @property
    def empty(self):
        return len(self.waiting) == 0 and len(self.running) == 0
