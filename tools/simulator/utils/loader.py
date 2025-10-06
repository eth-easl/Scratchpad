import json
from core.request import GenerationRequest
from core.arrival import PoissonProcess

def load_trace(
        trace_file,
        arrival_rate=None,
        filter_invalid=True,
        override_model=None,
    ):
    """
    Load requests from a trace file and optionally synthesize arrival times.

    Args:
        trace_file: Path to JSONL file containing request data
        arrival_rate: Optional arrival rate for Poisson process (requests/second)
        filter_invalid: Whether to filter out invalid requests (zero or negative lengths)
        override_model: Optional model name to override the model specified in trace

    Returns:
        List[GenerationRequest]: Loaded and processed requests
    """
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
        if override_model is not None:
            x["model"] = override_model
        input_len = x["reported_token_input"]
        output_len = x["reported_token_output"]
        request = GenerationRequest(
            f"{idx}", x["model"], input_len, output_len, workload[idx]
        )
        if filter_invalid and (input_len <= 0 or output_len <= 0):
            continue
        requests.append(request)
    print(f"Loaded {len(requests)} requests from {trace_file}")
    return requests