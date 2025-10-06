import json
from tools.simulator.core.request import GenerationRequest
from tools.benchmark.arrival import PoissonProcess

def load_trace(trace_file, arrival_rate=None):
    requests = []
    with open(trace_file, "r") as f:
        data = [json.loads(x) for x in f.readlines()]
    if arrival_rate is not None and arrival_rate > 0:
        print(f"Synthesizing arrival time using Poisson process with ar {arrival_rate}")
        pp = PoissonProcess(arrival_rate)
        duration_needed = len(data) / arrival_rate
        workload = pp.generate_workload(
            0, duration_needed * 2
        )  # multiply by 2 to be safe
    else:
        print("Arrival rate not provided, assuming all requests arrive at time 0")
        workload = [0] * len(data)
    for idx, x in enumerate(data):
        x["model"] = "meta-llama/Llama-2-7b-hf"
        if not len(x["input"]) == 0 and not len(x["output"]) == 0:
            request = GenerationRequest(
                f"{idx}", x["model"], len(x["input"]), len(x["output"]), workload[idx]
            )
            requests.append(request)
    return requests