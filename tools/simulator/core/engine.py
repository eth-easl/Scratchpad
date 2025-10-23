from collections import deque
from typing import Any, Deque, Dict, List, Optional

from internal.analyzer import ModelAnalyzer
from internal.configs.hardware_params import hardware_params

from .memory_planner import MemoryPlanner, ParallelConfig
from .request import GenerationRequest
from .trace import TraceEvent


class LLMEngine:
    def __init__(
        self,
        engine_id,
        model_name,
        hardware_name,
        w_bit,
        a_bit,
        kv_bit,
        *,
        parallel_config: Optional[ParallelConfig] = None,
        memory_override_bytes: Optional[float] = None,
    ):
        """
        Initialize a single LLM inference engine.

        Args:
            engine_id: Unique identifier for this engine instance
            model_name: Name of the model to be served (e.g., "meta-llama/Llama-2-7b-hf")
            hardware_name: Hardware platform name (e.g., "nvidia_A100")
            w_bit: Weight precision in bits (e.g., 16 for FP16, 8 for INT8)
            a_bit: Activation precision in bits
            kv_bit: KV cache precision in bits
            parallel_config: Optional tensor/pipeline parallel configuration
            memory_override_bytes: Override for device memory (bytes) per shard
        """
        self.engine_id = engine_id
        self.model_name = model_name
        self.hardware_name = hardware_name
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.kv_bit = kv_bit
        self.analyzer = ModelAnalyzer(
            model_id=model_name,
            hardware=hardware_name,
            config_file="internal/configs/llama.py",
            source="huggingface",
        )
        self.waiting: Deque[GenerationRequest] = deque()
        self.running: Deque[GenerationRequest] = deque()
        self.finished: List[GenerationRequest] = []
        self.failed: List[GenerationRequest] = []
        base_hardware = hardware_params.get(hardware_name)
        if base_hardware is None:
            raise ValueError(f"Unknown hardware profile: {hardware_name}")
        self.hardware_spec = dict(base_hardware)
        if memory_override_bytes is not None and memory_override_bytes > 0:
            self.hardware_spec["vmemory"] = max(
                memory_override_bytes, self.hardware_spec.get("vmemory", 0)
            )
        self.memory_override_bytes = memory_override_bytes
        self.memory_planner = MemoryPlanner(
            self.analyzer.model_params,
            self.hardware_spec,
            w_bit,
            a_bit,
            kv_bit,
            parallel_config=parallel_config,
        )
        self.parallel_config: ParallelConfig = self.memory_planner.parallel_config
        self.finished_requests: int = 0
        self.configure()

    def configure(self):
        """
        Configure engine-specific parameters. Override this method in subclasses
        to implement custom engine configurations.
        """
        pass

    def update_parallel_config(self, parallel_config: ParallelConfig) -> None:
        """
        Update tensor/pipeline parallel configuration and rebuild memory planner.

        Args:
            parallel_config: New parallel configuration to apply.
        """
        self.parallel_config = parallel_config
        self.memory_planner = MemoryPlanner(
            self.analyzer.model_params,
            self.hardware_spec,
            self.w_bit,
            self.a_bit,
            self.kv_bit,
            parallel_config=parallel_config,
        )

    def add_request(self, request: GenerationRequest):
        """
        Add a new request to the waiting queue.

        Args:
            request: The GenerationRequest to be added to the queue
        """
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
        decode_time = []
        finished_requests_in_this_batch = []
        executable_requests = []
        for req in requests:
            if self.memory_planner.can_allocate_request(req):
                self.memory_planner.allocate(req)
                executable_requests.append(req)
        batch_size = len(executable_requests)
        memory_event = self.memory_event(start_at)
        if batch_size == 0:
            return start_at + 0.0001, [], memory_event, []
        for req in executable_requests:
            if start_at < req.arrive_at:
                start_at = req.arrive_at
            decode_result = self.analyzer.analyze(
                req.input_length + req.generated_tokens,
                batchsize=batch_size,
                w_bit=self.w_bit,
                a_bit=self.a_bit,
                kv_bit=self.kv_bit,
            )
            decode_time.append(
                decode_result["total_results"]["decode"]["inference_time"]
            )
        finished_at = max(decode_time) + start_at
        finished_lst = []
        for req in executable_requests:
            finished = req._decode()  # Check if the request is finished
            if finished:
                req.set_generation_finished_at(finished_at)
                self.finished_requests += 1
                self.running.remove(req)
                self.finished.append(req)
                finished_requests_in_this_batch.append(req.req_id)
                finished_lst.append(req)
        self.memory_planner.free(finished_requests_in_this_batch)
        return finished_at, executable_requests, memory_event, finished_lst

    def step(self, start_at: float):
        """
        Execute one simulation step for this engine.

        This method implements the core scheduling logic:
        1. If there are waiting requests and memory is available, process prefill for one request
        2. If there are running requests, process decode phase for all of them
        3. If no requests are available, advance time slightly

        Args:
            start_at: The current simulation time in seconds

        Returns:
            tuple: (events, finished_requests, next_time, memory_event)
                - events: List of trace events for this step (or None)
                - finished_requests: List of requests that completed in this step
                - next_time: The next time this engine should be processed
                - memory_event: Memory usage event for tracing
        """
        # let's assume that process one request per step is fine in terms of utilization
        # self.memory_planner.print_status()

        if len(self.waiting) > 0:
            # Try to find a request that can be allocated
            allocatable_request = None
            request_to_remove = None

            # Look through waiting queue to find a request that fits in memory
            for i, req in enumerate(self.waiting):
                if self.memory_planner.can_allocate_request(req):
                    allocatable_request = req
                    request_to_remove = i
                    break

            if allocatable_request:
                # Remove the request from the queue and process it
                self.waiting.remove(allocatable_request)
                prefill_end_at, handled_requests, memory_event = self._prefill(
                    allocatable_request, start_at
                )
                return (
                    self.create_event(
                        "prefill", handled_requests, start_at, prefill_end_at
                    ),
                    [],
                    prefill_end_at,
                    memory_event,
                )
            else:
                # No requests can be allocated due to memory constraints
                # Move to the time step so simulation can progress
                return None, [], start_at + 0.0001, None

        elif len(self.running) > 0:
            # if there's no request needs prefill, proceed to decode
            # TODO(xiaozhe): let's assume we could do infinite batch size...
            (
                decode_finished_at,
                handled_requests,
                memory_event,
                finished_lst,
            ) = self._decode(list(self.running), start_at)
            return (
                self.create_event(
                    "decode", handled_requests, start_at, decode_finished_at
                ),
                finished_lst,
                decode_finished_at,
                memory_event,
            )
        else:
            # No work pending; signal that the engine can stay idle until new requests arrive
            return None, [], None, None

    def create_event(self, phase, handled_requests, start_at, end_at):
        """
        Create Chrome trace format events for performance visualization.

        Args:
            phase: Either "prefill" or "decode"
            handled_requests: List of requests processed in this phase
            start_at: Start time in seconds
            end_at: End time in seconds

        Returns:
            List of TraceEvent objects compatible with Chrome tracing format
        """
        complete_events = []
        start_us = int(max(start_at, 0) * 1_000_000)
        duration_s = max(end_at - start_at, 0.0)
        duration_us = max(int(duration_s * 1_000_000), 1)

        for req in handled_requests:
            event_args = {
                "request_id": req.req_id,
                "requested_model": req.model,
                "engine_id": str(self.engine_id),
                "engine_model": self.model_name,
                "hardware": self.hardware_name,
                "phase": phase,
                "start_time_s": round(start_at, 6),
                "end_time_s": round(end_at, 6),
                "duration_s": round(duration_s, 6),
            }

            if phase == "prefill":
                event_args.update(
                    {
                        "prompt_tokens": req.input_length,
                        "target_output_tokens": req.output_length,
                    }
                )
            elif phase == "decode":
                event_args.update(
                    {
                        "target_output_tokens": req.output_length,
                        "generated_tokens_total": req.generated_tokens,
                        "tokens_emitted_this_step": 1,
                    }
                )

            complete_events.append(
                TraceEvent(
                    name=f"{phase.upper()[0]}:{req.req_id}",
                    cat=f"request.{phase}",
                    ph="X",
                    pid=str(self.engine_id),
                    tid=0,
                    ts=start_us,
                    dur=duration_us,
                    args=event_args,
                )
            )

        return complete_events

    def memory_event(self, start_at):
        """
        Create a memory usage counter event for tracing.

        Args:
            start_at: Timestamp in seconds

        Returns:
            TraceEvent representing current memory block usage
        """
        used_blocks, total_blocks = self.memory_planner.usage()
        return TraceEvent(
            name="Memory usage",
            ph="C",
            ts=int(max(start_at, 0) * 1_000_000),
            pid=str(self.engine_id),
            tid=0,
            cat="memory.allocator",
            args={
                "used_blocks": used_blocks,
                "free_blocks": total_blocks - used_blocks,
                "engine_model": self.model_name,
                "hardware": self.hardware_name,
                "tensor_parallel_size": self.parallel_config.tensor_parallel_size,
                "pipeline_parallel_size": self.parallel_config.pipeline_parallel_size,
                "waiting_requests": [req.req_id for req in self.waiting],
                "running_requests": [req.req_id for req in self.running],
            },
        )

    def status_snapshot(self) -> Dict[str, Any]:
        """Return a concise status summary for periodic logging."""
        used_blocks, total_blocks = self.memory_planner.usage()
        if len(self.running) > 0:
            state = "busy"
        elif len(self.waiting) > 0:
            state = "queued"
        else:
            state = "idle"

        return {
            "engine_id": str(self.engine_id),
            "model": self.model_name,
            "hardware": self.hardware_name,
            "tensor_parallel_size": self.parallel_config.tensor_parallel_size,
            "state": state,
            "used_blocks": used_blocks,
            "total_blocks": total_blocks,
            "waiting": len(self.waiting),
            "running": len(self.running),
        }

    @property
    def empty(self):
        """
        Check if this engine has no requests to process.

        Returns:
            bool: True if both waiting and running queues are empty
        """
        return len(self.waiting) == 0 and len(self.running) == 0

    def reconfigure_model(
        self,
        model_name: str,
        *,
        w_bit: Optional[int] = None,
        a_bit: Optional[int] = None,
        kv_bit: Optional[int] = None,
    ) -> None:
        """Retarget this engine to serve a different model."""
        if self.waiting or self.running:
            raise RuntimeError("Cannot reconfigure engine while requests are in-flight")

        if w_bit is not None:
            self.w_bit = w_bit
        if a_bit is not None:
            self.a_bit = a_bit
        if kv_bit is not None:
            self.kv_bit = kv_bit

        self.model_name = model_name
        self.analyzer = ModelAnalyzer(
            model_id=model_name,
            hardware=self.hardware_name,
            config_file="internal/configs/llama.py",
            source="huggingface",
        )
        self.memory_planner = MemoryPlanner(
            self.analyzer.model_params,
            self.hardware_spec,
            self.w_bit,
            self.a_bit,
            self.kv_bit,
            parallel_config=self.parallel_config,
        )
        self.parallel_config = self.memory_planner.parallel_config
