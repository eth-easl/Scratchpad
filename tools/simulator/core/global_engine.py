import json
from collections import deque, defaultdict

from .engine import LLMEngine
from .request import GenerationRequest
from .policies import RandomGTLPolicy
from .policies.engine_reprovisioning import RandomReprovisioningPolicy
from .trace import TraceEvent
from core.env import EnvironmentChange, EnvironmentConfig, calculate_model_loading_time
from typing import Dict, Deque, List, Optional


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
        self.print_interval = print_interval
        self.last_print_time = 0.0

        # Initialize engines from environment configuration
        if environment_config:
            self._initialize_engines_from_config(environment_config)

    def _initialize_engines_from_config(self, config: EnvironmentConfig):
        """
        Initialize engines based on environment configuration.

        Args:
            config: Environment configuration containing GPU settings
        """
        print(f"Initializing engines from environment configuration...")

        total_loading_time = 0.0
        model_sizes_gb = {
            "meta-llama/Meta-Llama-3-70B-Instruct": 140.0,
        }

        for gpu_config in config.gpus:
            model_size_gb = model_sizes_gb.get(gpu_config.model, 140.0)

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
                    name=f"model-loading-{gpu_config.model}",
                    cat="model_loading",
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
                    name=f"model-preloaded-{gpu_config.model}",
                    cat="model_loading",
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
        self.timers[model_name][engine.engine_id] = 0

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
            name=f"engine-reprovision-{engine_to_repurpose.engine_id}",
            cat="engine_reprovisioning",
            ph="X",
            pid=engine_to_repurpose.engine_id,
            tid=0,
            ts=int(current_time * 1_000_000),
            dur=int(reprovisioning_time * 1_000_000),
            args={
                "source_model": source_model,
                "target_model": model_name,
                "engine_id": engine_to_repurpose.engine_id,
                "hardware": engine_to_repurpose.hardware_name,
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
        time_queue = set()

        # Track if any progress was made
        previous_finished = 0

        while True:
            made_progress = False

            for model in self.supported_models:
                for engine in self.engines[model]:
                    if self.timers[model][engine.engine_id] <= self.global_timer:
                        event, finished_lst, next_time, memory_event = engine.step(
                            self.timers[model][engine.engine_id]
                        )
                        self.timers[model][engine.engine_id] = next_time
                        time_queue.add(next_time)
                        if event is not None:
                            self._trace.extend(event)
                            self._trace.append(memory_event)
                            if finished_lst:  # Any finished requests means progress
                                made_progress = True

            # Add re-provisioning completion times to time queue
            for engine_id, (
                end_time,
                _,
            ) in self.engine_reprovisioning_in_progress.items():
                time_queue.add(end_time)

            if time_queue:
                self.global_timer = min(time_queue)
                time_queue.remove(self.global_timer)

                # Apply environment changes that should occur by this time
                self.apply_environment_changes(self.global_timer)

                # Check for completed re-provisioning
                reprov_completed = len(self.engine_reprovisioning_in_progress)
                self.check_reprovisioning_completions(self.global_timer)
                if len(self.engine_reprovisioning_in_progress) < reprov_completed:
                    made_progress = True  # Re-provisioning completed is progress

                self.check_new_requests(self.global_timer)

            # Update progress tracking
            current_finished = sum(
                engine.finished_requests
                for engines in self.engines.values()
                for engine in engines
            )
            if current_finished > previous_finished:
                made_progress = True
                self.last_progress_time = self.global_timer
                previous_finished = current_finished

            # Check for timeout - if no progress for too long, end simulation
            if made_progress:
                self.last_progress_time = self.global_timer
            elif self.global_timer - self.last_progress_time > self.simulation_timeout:
                print(
                    f"\nSimulation timeout: No progress for {self.simulation_timeout}s"
                )
                print(f"Finished: {self.finished_percentage:.2f}% of requests")
                break

            # Print progress with interval control
            if self.global_timer - self.last_print_time >= self.print_interval:
                print(
                    f"Finished: {self.finished_percentage:.2f}%, Current Time: {self.global_timer:.2f}, Engines: {self.total_engines}",
                    end="\r",
                )
                self.last_print_time = self.global_timer
            if not self.has_remaining_requests():
                break

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

    def check_new_requests(self, end_at):
        """
        Check for newly arrived requests and assign them to engines.

        Args:
            end_at: Current simulation time - requests with arrival_time <= end_at will be assigned
        """
        if len(self.pending_requests) > 0:
            allocatable_requests = [
                x for x in self.pending_requests if x.arrive_at <= end_at
            ]
            for req in allocatable_requests:
                print(
                    f"T: {self.global_timer:<.2f} Assigning request {req.req_id} @ {req.arrive_at:<.2f} to engine"
                )
                self.policy.assign_requests(req)
                self.pending_requests.remove(req)

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
