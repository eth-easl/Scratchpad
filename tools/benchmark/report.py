import os
import json
import uuid
from rich import box
from typing import Dict, List
from rich.table import Table
from rich.console import Console
from dataclasses import dataclass, fields, asdict
from tools.benchmark.common import BenchmarkMetrics, RequestFuncOutput


def print_benchmark(benchmark: BenchmarkMetrics):
    print(f"Benchmark Results")
    print(f"-" * 20)
    results = {}
    for field in fields(BenchmarkMetrics):
        key = field.name
        value = getattr(benchmark, field.name)
        if key.startswith("percentiles_"):
            for val in value:
                key = key.removeprefix("percentiles_")
                results[f"P{int(val[0])} {key}"] = round(val[1], 4)
        else:
            results[key] = round(value, 4)
    table = Table(title="Benchmark Results", box=box.SIMPLE)
    table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
    table.add_column("Value", justify="left", style="magenta")
    for key, value in results.items():
        table.add_row(key, str(value))
    console = Console()
    console.print(table)


def write_benchmark(
    benchmark: BenchmarkMetrics,
    output_dir: str,
    server_args: Dict,
    client_args: Dict,
    response_outputs: List[RequestFuncOutput],
):
    unique_id = str(uuid.uuid4())
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/benchmark_{unique_id}.jsonl"
    meta = {
        "server_args": server_args["system_info"],
        "client_args": vars(client_args),
        "metrics": asdict(benchmark),
    }
    with open(output_file, "w+") as f:
        f.write(json.dumps(meta) + "\n")
        for output in response_outputs:
            f.write(json.dumps(asdict(output)) + "\n")
    return output_file
