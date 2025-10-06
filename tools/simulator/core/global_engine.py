import json
from collections import deque, defaultdict

from .engine import LLMEngine
from .request import GenerationRequest
from .policies import RandomGTLPolicy
from .arrival import PoissonProcess

from typing import Dict, Deque

class LLMGlobalEngine:
    def __init__(self):
        self.engines = defaultdict(list[LLMEngine])
        self.timers = defaultdict(dict)
        self.pending_requests: Deque[GenerationRequest] = deque()
        self.global_timer = 0
        self.supported_models: set = set()
        self._trace = []
        self.total_requests = 0
        self.policy = RandomGTLPolicy()
        self.text2sql_requests: Dict[str, GenerationRequest] = {}
        self.arrival_rate = arrival_rate
        self.load_requests(input_file, arrival_rate)

    def add_engine(self, model_name, hardware_name, w_bit, a_bit, kv_bit):
        existing_engines = sum([len(x) for x in self.engines.values()])
        engine = LLMEngine(
            existing_engines + 1, model_name, hardware_name, w_bit, a_bit, kv_bit
        )
        self.engines[model_name].append(engine)
        self.supported_models.add(model_name)
        self.policy.prepare(self.engines)
        self.timers[model_name][engine.engine_id] = 0

    
    
    def handle_request_completion(self, request: GenerationRequest, current_time: float):
        if request.parent_request:
            parent = request.parent_request
            parent.update_stage(request, current_time)
            if parent.current_requests == [] and parent.current_stage < parent.total_stages:
                next_requests = parent.create_current_stage_requests("meta-llama/Llama-3.1-70B-Instruct", current_time)
                for next_request in next_requests:
                    if next_request is not None:
                        self.pending_requests.append(next_request)
    
    def SLO_pass_rate(self, SLO):
        pass_rate = 0
        for req_id, request in self.text2sql_requests.items():
            if request.current_stage < request.total_stages:
                continue
            else:
                if request.total_time <= SLO:
                    pass_rate += 1
        return pass_rate / len(self.text2sql_requests)

    def save_results(self, output_file: str):
        results = {}
        for req_id, request in self.text2sql_requests.items():
            if request.current_stage < request.total_stages:
                results.update({req_id: -1})
            else:
                results.update({req_id: request.total_time})
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

    def start(self):
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
                        if finished_lst:
                            for req in finished_lst:
                                self.handle_request_completion(req, next_time)
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
        if self.pending_requests:
            return True
        for model in self.supported_models:
            for engine in self.engines[model]:
                if len(engine.waiting) > 0 or len(engine.running) > 0:
                    return True
        return False

    def check_new_requests(self, end_at):
        if len(self.pending_requests) > 0:
            allocatable_requests = [x for x in self.pending_requests if x.arrive_at <= end_at]
            for req in allocatable_requests:
                # print(f"T: {self.global_timer} Assigning request {req.req_id} @ {req.arrive_at} to engine")
                self.policy.assign_requests(req)
                self.pending_requests.remove(req)

    @property
    def finished_percentage(self):
        total_finished = 0
        for model in self.supported_models:
            for engine in self.engines[model]:
                total_finished += engine.finished_requests
        return 100 * total_finished / self.total_requests

    @property
    def trace(self):
        return self._trace

    @property
    def requests_stats(self):
        stats = []
        for model in self.supported_models:
            for engine in self.engines[model]:
                stats.extend([x.to_dict() for x in engine.finished])
        return stats

    @property
    def config(self):
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
        failed_requests = []
        for model in self.supported_models:
            for engine in self.engines[model]:
                failed_requests.extend([x.to_dict() for x in engine.failed])
        return failed_requests