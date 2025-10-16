import os
import json
from dataclasses import asdict
from rich.console import Console
from core.global_engine import LLMGlobalEngine
from core.node_global_engine import NodeGlobalEngine
from utils.loader import load_trace
from core.env import load_environment_config, load_environment_changes

console = Console()


def run_simulation(args):
    print(args)
    workload = load_trace(
        args.input,
        float(args.arrival_rate),
    )
    if args.limit > 0:
        workload = workload[: args.limit]

    # Load environment configuration
    environment_config = None
    if hasattr(args, "environment") and args.environment:
        environment_config = load_environment_config(args.environment)

    # Load environment changes if provided
    environment_changes = None
    if hasattr(args, "environment_change_file") and args.environment_change_file:
        environment_changes = load_environment_changes(args.environment_change_file)

    # Choose engine type based on whether environment config is provided
    if environment_config:
        # Use NodeGlobalEngine when environment config is provided
        print("Using Node-based Global Engine")
        server = NodeGlobalEngine(
            environment_config=environment_config,
            environment_changes=environment_changes,
            print_interval=args.print_interval,
        )
    else:
        # Fallback to legacy LLMGlobalEngine for backward compatibility
        print("Using Legacy Engine-based Global Engine")
        server = LLMGlobalEngine(
            environment_config=environment_config,
            environment_changes=environment_changes,
            print_interval=args.print_interval,
        )

        # If no environment config is provided, use the old method
        for _ in range(args.n_engines):
            server.add_engine(
                "meta-llama/Meta-Llama-3-70B-Instruct", "nvidia_A100", 4, 4, 4
            )

    server.load_requests(workload)
    print(f"--" * 10 + " Simulation Started " + "--" * 10)
    server.start()

    # Collect stats (works for both legacy and node-based engines)
    if hasattr(server, "requests_stats"):
        summary = server.requests_stats
    else:
        summary = []

    if hasattr(server, "failed_requests"):
        failed = server.failed_requests
    else:
        failed = []

    if hasattr(server, "config"):
        config = server.config
    else:
        config = {"engine_type": "node_based" if environment_config else "legacy"}

    stats = {
        "summary": summary,
        "failed": failed,
        "config": config,
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
    parser.add_argument(
        "--environment",
        type=str,
        help="JSON file containing initial environment configuration (nodes, GPUs, bandwidth, etc.)",
        default=None,
    )
    parser.add_argument(
        "--environment-change-file",
        type=str,
        help="JSONL file containing dynamic environment changes (timestamp, gpu_name, amount)",
        default=None,
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        help="Print interval for progress updates in seconds (default: 0.1)",
        default=0.1,
    )
    args = parser.parse_args()
    run_simulation(args)
