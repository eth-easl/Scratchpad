import heapq
import json
import time
from collections import defaultdict, deque
from itertools import count
from typing import Deque, Dict, List, Optional

from core.env import EnvironmentChange, EnvironmentConfig, calculate_model_loading_time
from utils.model_utils import get_model_size_gb

from .engine import LLMEngine
from .policies import RandomGTLPolicy
from .policies.engine_reprovisioning import RandomReprovisioningPolicy
from .request import GenerationRequest
from .trace import TraceEvent


class LLMGlobalEngine:
    def __init__(
        self,
        environment_config: Optional[EnvironmentConfig] = None,
        environment_changes: Optional[List[EnvironmentChange]] = None,
        print_interval: float = 0.1,
    ):
        """
        Initialize the global engine that orchestrates multiple LLM engines.

        The global engine manages request scheduling across multiple engines,
        handles timing synchronization, and collects performance statistics.

        Args:
            environment_config: Optional initial environment configuration
            environment_changes: Optional list of environment changes to apply during simulation
            print_interval: Interval for printing progress updates in seconds
        """
        self.engines = defaultdict(list[LLMEngine])
        self.timers = defaultdict(dict)
        self.pending_requests: Deque[GenerationRequest] = deque()
        self.global_timer = 0
        self.supported_models: set = set()
        self._trace = []
        self.total_requests = 0
        self.policy = RandomGTLPolicy()
        self.reprovisioning_policy = RandomReprovisioningPolicy()
        self.text2sql_requests: Dict[str, GenerationRequest] = {}

        # Event-driven scheduler state
        self._event_queue: list[tuple[float, int, LLMEngine]] = []
        self._event_counter = count()

        # Environment configuration
        self.environment_config = environment_config
        self.environment_changes = environment_changes or []
        self.next_env_change_idx = 0

        # Track engine re-provisioning
        self.engine_reprovisioning_in_progress = (
            {}
        )  # engine_id -> (end_time, new_model)

        # Track simulation timeout for stuck scenarios
        self.last_progress_time = 0.0
        self.simulation_timeout = 300.0  # 5 minutes timeout
        self.print_interval = max(0.0, print_interval)
        self.last_print_time = 0.0
        self.last_print_wall_time = time.monotonic()

        # Initialize engines from environment configuration
        if environment_config:
            self._initialize_engines_from_config(environment_config)

    def _record_instant_event(
        self,
        name: str,
        category: str,
        timestamp_s: float,
        *,
        pid: str | int = "system",
        tid: str | int = 0,
        args: Optional[dict] = None,
    ) -> None:
        """Capture an instant-style trace event to aid human readability."""
        self._trace.append(
            TraceEvent(
                name=name,
                cat=category,
                ph="X",
                pid=pid,
                tid=tid,
                ts=int(max(timestamp_s, 0.0) * 1_000_000),
                dur=0,
                args=args or {},
            )
        )

    def record_assignment_event(
        self,
        request: GenerationRequest,
        engine: LLMEngine,
        assigned_at: float,
    ) -> None:
        """Log when a request is routed to a specific engine."""
        self._record_instant_event(
            name=f"Assigned :: request {request.req_id}",
            category="request.assignment",
            timestamp_s=assigned_at,
            pid=str(engine.engine_id),
            args={
                "request_id": request.req_id,
                "model": request.model,
                "engine_id": str(engine.engine_id),
                "engine_model": engine.model_name,
                "hardware": engine.hardware_name,
                "arrival_time_s": round(request.arrive_at, 6),
                "queued_duration_s": round(
                    max(assigned_at - request.arrive_at, 0.0), 6
                ),
            },
        )

    def record_completion_event(
        self,
        request: GenerationRequest,
        engine: LLMEngine,
    ) -> None:
        """Log when a request finishes generation on an engine."""
        finished_at = request.generation_finished_at or self.global_timer
        args = {
            "request_id": request.req_id,
            "model": request.model,
            "engine_id": str(engine.engine_id),
            "engine_model": engine.model_name,
            "hardware": engine.hardware_name,
            "arrival_time_s": round(request.arrive_at, 6),
            "completed_at_s": round(finished_at, 6),
            "latency_s": round(finished_at - request.arrive_at, 6),
            "output_tokens_generated": request.generated_tokens,
        }
        if request.prefill_finished_at is not None:
            args["prefill_finished_at_s"] = round(request.prefill_finished_at, 6)
        self._record_instant_event(
            name=f"Completed :: request {request.req_id}",
            category="request.lifecycle",
            timestamp_s=finished_at,
            pid=str(engine.engine_id),
            args=args,
        )

    def _reset_event_queue(self) -> None:
        """Clear queued engine events and reset the sequence counter."""
        self._event_queue.clear()
        self._event_counter = count()

    def _schedule_engine(self, engine: LLMEngine, ready_time: float) -> None:
        """
        Schedule an engine to take its next step at the provided simulation time.

        Args:
            engine: Engine instance that has pending work
            ready_time: Simulation timestamp when the engine should be revisited
        """
        self.timers[engine.model_name][engine.engine_id] = ready_time
        heapq.heappush(
            self._event_queue,
            (ready_time, next(self._event_counter), engine),
        )

    def _process_engine_events(self) -> bool:
        """
        Execute all engine steps scheduled for the current simulation time.

        Returns:
            bool: True if any request completed or new trace events were recorded
        """
        progress_made = False
        while self._event_queue and self._event_queue[0][0] <= self.global_timer:
            event_time, _, engine = heapq.heappop(self._event_queue)
            scheduled_time = self.timers[engine.model_name].get(engine.engine_id)

            # Skip stale events that were superseded by a newer schedule
            if scheduled_time is None or abs(event_time - scheduled_time) > 1e-9:
                continue

            # Advance simulation time to the event if needed
            if event_time > self.global_timer:
                self.global_timer = event_time

            progress_made = True
            event, finished_lst, next_time, memory_event = engine.step(event_time)

            if event:
                self._trace.extend(event)
            if memory_event is not None:
                self._trace.append(memory_event)

            if finished_lst:
                progress_made = True
                for finished_request in finished_lst:
                    self.record_completion_event(finished_request, engine)
                    self.handle_request_completion(finished_request, event_time)

            if next_time is not None:
                self._schedule_engine(engine, next_time)
            else:
                self.timers[engine.model_name][engine.engine_id] = None

        return progress_made

    def _next_pending_request_arrival(self, after_time: float) -> Optional[float]:
        """
        Find the earliest arrival time among pending requests that is after the provided time.
        """
        next_time: Optional[float] = None
        for req in self.pending_requests:
            if req.arrive_at > after_time:
                if next_time is None or req.arrive_at < next_time:
                    next_time = req.arrive_at
        return next_time

    def _next_environment_change_time(self, after_time: float) -> Optional[float]:
        """
        Fetch the timestamp of the next scheduled environment change.
        """
        if self.next_env_change_idx < len(self.environment_changes):
            change_time = self.environment_changes[self.next_env_change_idx].timestamp
            if change_time > after_time:
                return change_time
        return None

    def _next_reprovision_completion_time(self, after_time: float) -> Optional[float]:
        """
        Determine the earliest completion time among in-flight re-provisioning operations.
        """
        next_time: Optional[float] = None
        for end_time, _ in self.engine_reprovisioning_in_progress.values():
            if end_time > after_time:
                if next_time is None or end_time < next_time:
                    next_time = end_time
        return next_time

    def on_request_assigned(
        self, engine: LLMEngine, *, scheduled_time: Optional[float] = None
    ) -> None:
        """
        Notify the global engine that a request was assigned to an engine.

        Args:
            engine: Engine that received new work
            scheduled_time: Optional timestamp to use instead of the current time
        """
        engine_timers = self.timers[engine.model_name]
        if engine_timers.get(engine.engine_id) is None:
            ready_time = (
                scheduled_time if scheduled_time is not None else self.global_timer
            )
            self._schedule_engine(engine, ready_time)

    def _initialize_engines_from_config(self, config: EnvironmentConfig):
        """
        Initialize engines based on environment configuration.

        Args:
            config: Environment configuration containing GPU settings
        """
        print(f"Initializing engines from environment configuration...")

        total_loading_time = 0.0

        for gpu_config in config.gpus:
            model_size_gb = get_model_size_gb(
                gpu_config.model,
                weight_bits=gpu_config.precision.weight_bits,
            )

            # Calculate loading time if model is not preloaded
            loading_time = 0.0
            if not config.model_loading.preload_models:
                loading_time = calculate_model_loading_time(
                    model_size_gb=model_size_gb,
                    disk_bandwidth_mbps=config.infrastructure.disk_bandwidth_mbps,
                    pcie_bandwidth_gbps=config.infrastructure.pcie_bandwidth_gbps,
                    cache_available=False,
                )
                total_loading_time = max(total_loading_time, loading_time)
                print(f"  Model {gpu_config.model} loading time: {loading_time:.2f}s")

                # Add model loading trace event
                loading_event = TraceEvent(
                    name=f"Load :: {gpu_config.model}",
                    cat="system.model_loading",
                    ph="X",
                    pid=-1,  # Use -1 for system-level events
                    tid=0,
                    ts=0,  # Start at time 0
                    dur=int(loading_time * 1_000_000),  # Convert to microseconds
                    args={
                        "model": gpu_config.model,
                        "hardware": gpu_config.name,
                        "model_size_gb": model_size_gb,
                        "disk_bandwidth_mbps": config.infrastructure.disk_bandwidth_mbps,
                        "pcie_bandwidth_gbps": config.infrastructure.pcie_bandwidth_gbps,
                        "loading_time_s": loading_time,
                        "engines_created": gpu_config.amount,
                    },
                )
                self._trace.append(loading_event)
            else:
                print(f"  Model {gpu_config.model} is preloaded")

                # Add preloaded model trace event
                loading_event = TraceEvent(
                    name=f"Preloaded :: {gpu_config.model}",
                    cat="system.model_loading",
                    ph="X",
                    pid=-1,
                    tid=0,
                    ts=0,
                    dur=0,  # Instant for preloaded models
                    args={
                        "model": gpu_config.model,
                        "hardware": gpu_config.name,
                        "model_size_gb": model_size_gb,
                        "preloaded": True,
                        "engines_created": gpu_config.amount,
                    },
                )
                self._trace.append(loading_event)

            # Add the specified number of engines
            for i in range(gpu_config.amount):
                self.add_engine(
                    gpu_config.model,
                    gpu_config.name,
                    gpu_config.precision.weight_bits,
                    gpu_config.precision.activation_bits,
                    gpu_config.precision.kv_bits,
                )

        if total_loading_time > 0:
            print(f"Total initialization time: {total_loading_time:.2f}s")
            self.global_timer = total_loading_time  # Start simulation after loading
            print(f"Simulation starts at time: {self.global_timer:.2f}s")

    def add_engine(self, model_name, hardware_name, w_bit, a_bit, kv_bit):
        """
        Add a new LLM engine to the global engine pool.

        Args:
            model_name: Name of the model to be served by this engine
            hardware_name: Hardware platform for this engine
            w_bit: Weight precision in bits
            a_bit: Activation precision in bits
            kv_bit: KV cache precision in bits
        """
        existing_engines = sum([len(x) for x in self.engines.values()])
        engine = LLMEngine(
            existing_engines + 1, model_name, hardware_name, w_bit, a_bit, kv_bit
        )
        self.engines[model_name].append(engine)
        self.supported_models.add(model_name)
        self.policy.prepare(self.engines)
        self.policy.set_global_engine(self)
        self.timers[model_name][engine.engine_id] = None

    def add_engines_at_time(
        self,
        model_name: str,
        hardware_name: str,
        w_bit: int,
        a_bit: int,
        kv_bit: int,
        amount: int,
    ):
        """
        Add multiple engines of the same configuration.

        Args:
            model_name: Name of the model to be served by these engines
            hardware_name: Hardware platform for these engines
            w_bit: Weight precision in bits
            a_bit: Activation precision in bits
            kv_bit: KV cache precision in bits
            amount: Number of engines to add
        """
        for _ in range(amount):
            self.add_engine(model_name, hardware_name, w_bit, a_bit, kv_bit)

    def apply_environment_changes(self, current_time: float):
        """
        Apply any pending environment changes that should occur by the given time.

        Args:
            current_time: Current simulation time in seconds
        """
        while self.next_env_change_idx < len(self.environment_changes):
            env_change = self.environment_changes[self.next_env_change_idx]

            if env_change.timestamp > current_time:
                break

            # Apply the environment change
            print(
                f"T: {current_time:.2f} Adding {env_change.amount} {env_change.gpu_name} engines"
            )

            # Add environment change trace event
            env_event = TraceEvent(
                name=f"env-change-{env_change.gpu_name}",
                cat="environment_change",
                ph="X",
                pid=-1,  # Use -1 for system-level events
                tid=0,
                ts=int(current_time * 1_000_000),  # Convert to microseconds
                dur=0,  # Instant event
                args={
                    "gpu_name": env_change.gpu_name,
                    "amount": env_change.amount,
                    "timestamp": env_change.timestamp,
                    "total_engines_after": self.total_engines + env_change.amount,
                },
            )
            self._trace.append(env_event)

            # Use default precision settings - these could be made configurable later
            self.add_engines_at_time(
                "meta-llama/Meta-Llama-3-70B-Instruct",
                env_change.gpu_name,
                4,
                4,
                4,
                env_change.amount,
            )

            self.next_env_change_idx += 1

    def can_create_engine_for_model(self, model_name: str) -> bool:
        """
        Check if we can create an engine for the given model (either by re-provisioning or creating new).

        Args:
            model_name: The model to create an engine for

        Returns:
            bool: True if an engine can be created for the model
        """
        # Check if we have existing engines that can be re-purposed
        for model, engines in self.engines.items():
            for engine in engines:
                # Skip engines that are already being re-provisioned
                if engine.engine_id in self.engine_reprovisioning_in_progress:
                    continue
                if self.reprovisioning_policy.can_repurpose_engine(
                    engine, model_name, self.global_timer
                ):
                    return True
        return False

    def create_engine_for_model(
        self, model_name: str, current_time: float
    ) -> Optional[float]:
        """
        Create an engine for the given model by re-provisioning an existing idle engine.

        Args:
            model_name: The model to create an engine for
            current_time: Current simulation time

        Returns:
            Optional[float]: Time when the engine will be ready, or None if creation failed
        """
        # Try to find an engine to re-purpose
        engine_to_repurpose = self.reprovisioning_policy.select_engine_to_repurpose(
            model_name,
            self.engines,
            current_time,
            set(self.engine_reprovisioning_in_progress.keys()),
        )

        if engine_to_repurpose is None:
            return None

        # Calculate re-provisioning time
        source_model = engine_to_repurpose.model_name
        reprovisioning_time = self.reprovisioning_policy.get_reprovisioning_time(
            source_model, model_name, engine_to_repurpose
        )

        # Mark engine as being re-provisioned
        self.engine_reprovisioning_in_progress[engine_to_repurpose.engine_id] = (
            current_time + reprovisioning_time,
            model_name,
        )

        # Add re-provisioning trace event
        reprovision_event = TraceEvent(
            name=f"Engine {engine_to_repurpose.engine_id} reprovision :: {model_name}",
            cat="engine.reprovisioning",
            ph="X",
            pid=str(engine_to_repurpose.engine_id),
            tid=0,
            ts=int(current_time * 1_000_000),
            dur=int(reprovisioning_time * 1_000_000),
            args={
                "engine_id": str(engine_to_repurpose.engine_id),
                "hardware": engine_to_repurpose.hardware_name,
                "source_model": source_model,
                "target_model": model_name,
                "reprovisioning_time_s": reprovisioning_time,
            },
        )
        self._trace.append(reprovision_event)

        print(
            f"T: {current_time:.2f} Re-provisioning engine {engine_to_repurpose.engine_id} "
            f"from {source_model} to {model_name} (takes {reprovisioning_time:.2f}s)"
        )

        return current_time + reprovisioning_time

    def check_reprovisioning_completions(self, current_time: float):
        """
        Check if any engine re-provisioning has completed and update the engines.

        Args:
            current_time: Current simulation time
        """
        completed_reprovisioning = []

        for engine_id, (
            end_time,
            new_model,
        ) in self.engine_reprovisioning_in_progress.items():
            if current_time >= end_time:
                completed_reprovisioning.append((engine_id, new_model))

        for engine_id, new_model in completed_reprovisioning:
            # Find and update the engine
            for old_model, engines in self.engines.items():
                for i, engine in enumerate(engines):
                    if engine.engine_id == engine_id:
                        # Remove from old model list
                        engines.remove(engine)
                        if len(engines) == 0:
                            del self.engines[old_model]
                        # Clean timer entry for previous model
                        self.timers[old_model].pop(engine.engine_id, None)

                        # Update engine properties
                        engine.model_name = new_model
                        engine.waiting.clear()
                        engine.running.clear()
                        engine.finished.clear()
                        engine.failed.clear()

                        # Create new memory planner for the new model
                        from internal.configs.hardware_params import hardware_params

                        engine.memory_planner = (
                            engine.analyzer.memory_planner.__class__(
                                engine.analyzer.model_params,
                                hardware_params[engine.hardware_name],
                                engine.w_bit,
                                engine.a_bit,
                                engine.kv_bit,
                            )
                        )

                        # Add to new model list
                        if new_model not in self.engines:
                            self.engines[new_model] = []
                        self.engines[new_model].append(engine)
                        self.timers[new_model][engine.engine_id] = None

                        # Update supported models
                        self.supported_models.add(new_model)
                        if len(self.engines.get(old_model, [])) == 0:
                            self.supported_models.discard(old_model)

                        # Re-prepare policy
                        self.policy.prepare(self.engines)
                        self.policy.set_global_engine(self)

                        print(
                            f"T: {current_time:.2f} Engine {engine_id} re-provisioning completed, "
                            f"now serving {new_model}"
                        )
                        self._record_instant_event(
                            name=f"Engine {engine_id} ready :: {new_model}",
                            category="engine.reprovisioning",
                            timestamp_s=current_time,
                            pid=str(engine_id),
                            args={
                                "engine_id": str(engine_id),
                                "hardware": engine.hardware_name,
                                "new_model": new_model,
                                "previous_model": old_model,
                                "reprovision_completed_at_s": round(current_time, 6),
                            },
                        )
                        break

            # Remove from tracking
            del self.engine_reprovisioning_in_progress[engine_id]

    def load_requests(self, requests: List[GenerationRequest]):
        """
        Load a batch of requests into the pending queue.

        Args:
            requests: List of GenerationRequest objects to be processed
        """
        for req in requests:
            self.pending_requests.append(req)
            self.total_requests += 1
            self._record_instant_event(
                name=f"Request arrived :: {req.req_id}",
                category="request.lifecycle",
                timestamp_s=req.arrive_at,
                args={
                    "request_id": req.req_id,
                    "model": req.model,
                    "input_tokens": req.input_length,
                    "target_output_tokens": req.output_length,
                },
            )

    def handle_request_completion(
        self, request: GenerationRequest, current_time: float
    ):
        """
        Handle completion of a request, potentially triggering multi-stage workflows.

        Args:
            request: The completed request
            current_time: Current simulation time
        """
        if request.parent_request:
            parent = request.parent_request
            parent.update_stage(request, current_time)
            if (
                parent.current_requests == []
                and parent.current_stage < parent.total_stages
            ):
                next_requests = parent.create_current_stage_requests(
                    "meta-llama/Llama-3.1-70B-Instruct", current_time
                )
                for next_request in next_requests:
                    if next_request is not None:
                        self.pending_requests.append(next_request)

    def SLO_pass_rate(self, SLO):
        """
        Calculate the Service Level Objective (SLO) pass rate for text2sql requests.

        Args:
            SLO: Service Level Objective time threshold in seconds

        Returns:
            float: Percentage of requests that completed within the SLO time
        """
        pass_rate = 0
        for req_id, request in self.text2sql_requests.items():
            if request.current_stage < request.total_stages:
                continue
            else:
                if request.total_time <= SLO:
                    pass_rate += 1
        return pass_rate / len(self.text2sql_requests)

    def save_results(self, output_file: str):
        """
        Save simulation results to a JSON file.

        Args:
            output_file: Path to the output JSON file
        """
        results = {}
        for req_id, request in self.text2sql_requests.items():
            if request.current_stage < request.total_stages:
                results.update({req_id: -1})
            else:
                results.update({req_id: request.total_time})
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    def start(self):
        """
        Start the simulation loop. This is the main simulation driver.

        The simulation proceeds by:
        1. Processing engines that are ready to execute at the current global time
        2. Advancing global time to the next earliest event
        3. Applying any environment changes scheduled for this time
        4. Checking for completed engine re-provisioning
        5. Assigning newly arrived requests to engines
        6. Continuing until all requests are processed
        """
        print(f"Total requests: {self.total_requests}")
        if self.environment_changes:
            print(
                f"Environment changes: {len(self.environment_changes)} events scheduled"
            )
        self._reset_event_queue()
        previous_finished = 0
        self.last_progress_time = self.global_timer
        self.last_print_time = self.global_timer
        self.last_print_wall_time = time.monotonic()

        while True:
            progress_at_current_time = False

            # Drain all events that occur at the current simulation time
            while True:
                loop_progress = False

                prev_env_idx = self.next_env_change_idx
                self.apply_environment_changes(self.global_timer)
                if self.next_env_change_idx > prev_env_idx:
                    loop_progress = True

                reprov_before = len(self.engine_reprovisioning_in_progress)
                self.check_reprovisioning_completions(self.global_timer)
                if len(self.engine_reprovisioning_in_progress) < reprov_before:
                    loop_progress = True

                assigned_now = self.check_new_requests(self.global_timer)
                if assigned_now:
                    loop_progress = True

                if self._process_engine_events():
                    loop_progress = True

                if not loop_progress:
                    break

                progress_at_current_time = True

            # Update progress tracking
            current_finished = sum(
                engine.finished_requests
                for engines in self.engines.values()
                for engine in engines
            )
            if current_finished > previous_finished:
                previous_finished = current_finished
                progress_at_current_time = True

            if progress_at_current_time:
                self.last_progress_time = self.global_timer
            elif self.global_timer - self.last_progress_time > self.simulation_timeout:
                print(
                    f"\nSimulation timeout: No progress for {self.simulation_timeout}s"
                )
                print(f"Finished: {self.finished_percentage:.2f}% of requests")
                break

            # Print progress with interval control (supports wall and sim time)
            if self.print_interval > 0.0:
                now_wall = time.monotonic()
                wall_elapsed = now_wall - self.last_print_wall_time
                sim_elapsed = self.global_timer - self.last_print_time
                if (
                    wall_elapsed >= self.print_interval
                    or sim_elapsed >= self.print_interval
                ):
                    print(
                        f"Finished: {self.finished_percentage:.2f}%, Current Time: {self.global_timer:.2f}, Engines: {self.total_engines}",
                        end="\r",
                        flush=True,
                    )
                    self.last_print_time = self.global_timer
                    self.last_print_wall_time = now_wall

            if not self.has_remaining_requests():
                break

            candidate_times: List[float] = []
            if self._event_queue:
                candidate_times.append(self._event_queue[0][0])
            next_request = self._next_pending_request_arrival(self.global_timer)
            if next_request is not None:
                candidate_times.append(next_request)
            next_env_change = self._next_environment_change_time(self.global_timer)
            if next_env_change is not None:
                candidate_times.append(next_env_change)
            next_reprov = self._next_reprovision_completion_time(self.global_timer)
            if next_reprov is not None:
                candidate_times.append(next_reprov)

            if not candidate_times:
                break

            next_time = min(candidate_times)
            if next_time <= self.global_timer:
                next_time = self.global_timer + 1e-9

            self.global_timer = next_time

    def has_remaining_requests(self):
        """
        Check if there are any requests remaining to be processed.

        Returns:
            bool: True if there are pending or running requests across all engines
        """
        if self.pending_requests:
            return True
        for model in self.supported_models:
            for engine in self.engines[model]:
                if len(engine.waiting) > 0 or len(engine.running) > 0:
                    return True
        return False

    def check_new_requests(self, end_at) -> int:
        """
        Check for newly arrived requests and assign them to engines.

        Args:
            end_at: Current simulation time - requests with arrival_time <= end_at will be assigned
        """
        assigned_count = 0
        if len(self.pending_requests) > 0:
            allocatable_requests = [
                x for x in list(self.pending_requests) if x.arrive_at <= end_at
            ]
            for req in allocatable_requests:
                assigned = self.policy.assign_requests(req)
                if assigned:
                    print(
                        f"T: {self.global_timer:<.2f} Assigning request {req.req_id} @ {req.arrive_at:<.2f} to engine"
                    )
                    self.pending_requests.remove(req)
                    assigned_count += 1
        return assigned_count

    @property
    def total_engines(self):
        """
        Calculate the total number of engines across all models.

        Returns:
            int: Total number of engines
        """
        return sum(len(engines) for engines in self.engines.values())

    @property
    def finished_percentage(self):
        """
        Calculate the percentage of requests that have been completed.

        Returns:
            float: Percentage (0-100) of finished requests
        """
        total_finished = 0
        for model in self.supported_models:
            for engine in self.engines[model]:
                total_finished += engine.finished_requests
        return 100 * total_finished / self.total_requests

    @property
    def trace(self):
        """
        Get the Chrome trace format events for all processed requests.

        Returns:
            List[TraceEvent]: Events that can be loaded into Chrome tracing
        """
        return self._trace

    @property
    def requests_stats(self):
        """
        Get detailed statistics for all completed requests.

        Returns:
            List[dict]: Statistics including timing information for each completed request
        """
        stats = []
        for model in self.supported_models:
            for engine in self.engines[model]:
                stats.extend([x.to_dict() for x in engine.finished])
        return stats

    @property
    def config(self):
        """
        Get the configuration of all engines in the system.

        Returns:
            dict: Configuration details for each model and its engines
        """
        configuration = {}
        for model in self.supported_models:
            configuration[model] = {}
            engines = []
            for engine in self.engines[model]:
                engine_config = {
                    "model": model,
                    "hardware": engine.hardware_name,
                    "w_bit": engine.w_bit,
                    "a_bit": engine.a_bit,
                    "kv_bit": engine.kv_bit,
                }
                engines.append(engine_config)
            configuration[model]["engines"] = engines
        return configuration

    @property
    def summary(self):
        """
        Calculate performance summary statistics for all completed requests.

        Returns:
            List[dict]: Performance metrics including latency, throughput, and percentiles
        """
        stats = self.requests_stats

        avg_latency = sum(
            [x["generation_finished_at"] - x["arrive_at"] for x in stats]
        ) / len(stats)
        throughput = len(stats) / max([x["generation_finished_at"] for x in stats])
        p90_latency = sorted(
            [x["generation_finished_at"] - x["arrive_at"] for x in stats]
        )[int(len(stats) * 0.9)]
        p95_latency = sorted(
            [x["generation_finished_at"] - x["arrive_at"] for x in stats]
        )[int(len(stats) * 0.95)]
        avg_time_to_first_token = sum(
            [x["prefill_finished_at"] - x["arrive_at"] for x in stats]
        ) / len(stats)
        p90_ttft = sorted([x["prefill_finished_at"] - x["arrive_at"] for x in stats])[
            int(len(stats) * 0.9)
        ]
        p95_ttft = sorted([x["prefill_finished_at"] - x["arrive_at"] for x in stats])[
            int(len(stats) * 0.95)
        ]
        summaries = [
            {"Metric": "Avg E2E-Latency (s)", "Value": avg_latency},
            {"Metric": "Avg TTFT (s)", "Value": avg_time_to_first_token},
            {"Metric": "Throughput (req/s)", "Value": throughput},
            {"Metric": "P90 Latency (s)", "Value": p90_latency},
            {"Metric": "P95 Latency (s)", "Value": p95_latency},
            {"Metric": "P90 TTFT (s)", "Value": p90_ttft},
            {"Metric": "P95 TTFT (s)", "Value": p95_ttft},
        ]
        return summaries

    @property
    def failed_requests(self):
        """
        Get statistics for all failed requests.

        Returns:
            List[dict]: Failed request information for debugging and analysis
        """
        failed_requests = []
        for model in self.supported_models:
            for engine in self.engines[model]:
                failed_requests.extend([x.to_dict() for x in engine.failed])
        return failed_requests
