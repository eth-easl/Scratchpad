import json
from collections import deque, defaultdict

from .engine import LLMEngine
from .request import GenerationRequest
from .policies import RandomGTLPolicy
from typing import Dict, Deque, List

class LLMGlobalEngine:
    def __init__(self):
        """
        Initialize the global engine that orchestrates multiple LLM engines.

        The global engine manages request scheduling across multiple engines,
        handles timing synchronization, and collects performance statistics.
        """
        self.engines = defaultdict(list[LLMEngine])
        self.timers = defaultdict(dict)
        self.pending_requests: Deque[GenerationRequest] = deque()
        self.global_timer = 0
        self.supported_models: set = set()
        self._trace = []
        self.total_requests = 0
        self.policy = RandomGTLPolicy()
        self.text2sql_requests: Dict[str, GenerationRequest] = {}

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
        self.timers[model_name][engine.engine_id] = 0
    
    def load_requests(self, requests: List[GenerationRequest]):
        """
        Load a batch of requests into the pending queue.

        Args:
            requests: List of GenerationRequest objects to be processed
        """
        for req in requests:
            self.pending_requests.append(req)
            self.total_requests += 1

    def handle_request_completion(self, request: GenerationRequest, current_time: float):
        """
        Handle completion of a request, potentially triggering multi-stage workflows.

        Args:
            request: The completed request
            current_time: Current simulation time
        """
        if request.parent_request:
            parent = request.parent_request
            parent.update_stage(request, current_time)
            if parent.current_requests == [] and parent.current_stage < parent.total_stages:
                next_requests = parent.create_current_stage_requests("meta-llama/Llama-3.1-70B-Instruct", current_time)
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
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

    def start(self):
        """
        Start the simulation loop. This is the main simulation driver.

        The simulation proceeds by:
        1. Processing engines that are ready to execute at the current global time
        2. Advancing global time to the next earliest event
        3. Assigning newly arrived requests to engines
        4. Continuing until all requests are processed
        """
        print(f"Total requests: {self.total_requests}")
        time_queue = set()
        while True:
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
            if time_queue:
                self.global_timer = min(time_queue)
                time_queue.remove(self.global_timer)
                self.check_new_requests(self.global_timer)

            print(
                f"Finished: {self.finished_percentage:.2f}%, Current Time: {self.global_timer:.2f}",
                end="\r",
            )
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
            allocatable_requests = [x for x in self.pending_requests if x.arrive_at <= end_at]
            for req in allocatable_requests:
                print(f"T: {self.global_timer:<.2f} Assigning request {req.req_id} @ {req.arrive_at:<.2f} to engine")
                self.policy.assign_requests(req)
                self.pending_requests.remove(req)

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
                    'model': model,
                    'hardware': engine.hardware_name,
                    'w_bit': engine.w_bit,
                    'a_bit': engine.a_bit,
                    'kv_bit': engine.kv_bit,
                }
                engines.append(engine_config)
            configuration[model]['engines'] = engines
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