from rich.console import Console
from rich.table import Table
from rich import box
from tools.benchmark.common import BenchmarkMetrics
from dataclasses import dataclass, fields


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
