import os
import json
from dataclasses import asdict
from rich.console import Console
from core.global_engine import LLMGlobalEngine
from utils.loader import load_trace

console = Console()

def run_simulation(args):
    print(args)
    workload = load_trace(
        args.input,
        float(args.arrival_rate),
        override_model="meta-llama/Meta-Llama-3-70B-Instruct",
    )
    if args.limit > 0:
        workload = workload[: args.limit]
    server = LLMGlobalEngine()

    for i in range(args.n_engines):
        server.add_engine("meta-llama/Meta-Llama-3-70B-Instruct", "nvidia_A100", 4, 4, 4)

    server.load_requests(workload)

    server.start()
    
    stats = {
        "summary": server.requests_stats,
        "failed": server.failed_requests,
        "config": server.config,
    }
    os.makedirs(os.path.dirname(args.trace_output), exist_ok=True)
    os.makedirs(os.path.dirname(args.stats_output), exist_ok=True)
    with open(args.trace_output, "w") as f:
        data = {"traceEvents": [asdict(x) for x in server.trace]}
        f.write(json.dumps(data, indent=4))
    with open(args.stats_output, "w") as f:
        f.write(json.dumps(stats, indent=4))

    print(end="\n")
    print(f"--" * 10 + " Simulation Done " + "--" * 10)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input file")
    parser.add_argument("--n-engines", type=int, help="Number of engines")
    parser.add_argument("--arrival-rate", help="Arrival rate", default=None)
    parser.add_argument(
        "--trace-output",
        type=str,
        help="Trace file",
        default=".local/replay_results/trace.json",
    )
    parser.add_argument(
        "--stats-output",
        type=str,
        help="Stats file",
        default=".local/replay_results/stats.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of requests",
        default=-1,
    )
    args = parser.parse_args()
    run_simulation(args)