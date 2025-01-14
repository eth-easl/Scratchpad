import time
import argparse
from argparse import Namespace
import asyncio
import requests
from typing import List, Optional, Callable, Dict
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from tools.benchmark.common import (
    construct_dataset,
    get_request,
    async_request_openai_completions,
    async_request_sp_sysinfo,
    RequestFuncOutput,
    RequestFuncInput,
    calculate_metrics,
)
from tools.benchmark.report import print_benchmark, write_benchmark


def check_goodput_args(args: Namespace):
    # Check and parse goodput arguments
    gootput_config_dict = {}
    VALID_NAMES = ["ttft", "tpot", "e2el"]
    if args.goodput:
        gootput_config_dict = parse_goodput(args.goodput)
        for slo_name, slo_val in gootput_config_dict.items():
            if slo_name not in VALID_NAMES:
                raise ValueError(
                    f"Invalid metric name found, {slo_name}: {slo_val}. "
                    "The service level objective name should be one of "
                    f"{str(VALID_NAMES)}. "
                )
            if slo_val < 0:
                raise ValueError(
                    f"Invalid value found, {slo_name}: {slo_val}. "
                    "The service level objective value should be "
                    "non-negative."
                )
    return gootput_config_dict


def parse_goodput(slo_pairs):
    gootput_config_dict = {}
    try:
        for slo_pair in slo_pairs:
            slo_name, slo_val = slo_pair.split(":")
            gootput_config_dict[slo_name] = float(slo_val)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            "Invalid format found for service level objectives. "
            'Specify service level objectives for goodput as "KEY:VALUE" '
            "pairs, where the key is a metric name, and the value is a "
            "number in milliseconds."
        ) from err
    return gootput_config_dict


async def run_benchmark(
    args: Namespace,
    input_requests: List[RequestFuncInput],
    request_func: Callable,
    tokenizer: PreTrainedTokenizerBase,
    goodput_config_dict: Dict[str, float],
    max_concurrency: Optional[int] = None,
):
    system_info = await async_request_sp_sysinfo(args.endpoint)
    pbar = tqdm(total=len(input_requests))
    tasks: List[asyncio.Task] = []
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            return await request_func(request_func_input=request_func_input, pbar=pbar)
        async with semaphore:
            return await request_func(request_func_input=request_func_input, pbar=pbar)

    benchmark_start_time = time.perf_counter()
    async for request in get_request(input_requests, args.request_rate):
        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input=request, pbar=pbar)
            )
        )
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)
    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time
    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentile_metrics=args.percentile_metrics.split(","),
        selected_percentiles=[float(p) for p in args.metric_percentiles.split(",")],
        goodput_config_dict=goodput_config_dict,
    )
    print_benchmark(metrics)
    if args.output:
        output_file = write_benchmark(
            metrics,
            args.output,
            system_info,
            args,
            outputs,
        )
        print(f"Results written to {output_file}")
    return metrics


def benchmark(args):
    print(args)
    if args.model == "":
        args.model = args.tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    bench_requests = construct_dataset(
        args.endpoint, args.dataset, tokenizer, args.num_prompts
    )
    for req in bench_requests:
        req.model = args.model
    request_func = async_request_openai_completions
    gootput_config_dict = check_goodput_args(args)
    # check if server is ready
    server_ready = False
    if args.wait_until_ready:
        while not server_ready:
            try:
                requests.get(args.endpoint)
                server_ready = True
            except Exception as e:
                print("Server is not ready. Please start the server first.")
                time.sleep(5)
    asyncio.run(
        run_benchmark(
            args,
            bench_requests,
            request_func,
            tokenizer,
            goodput_config_dict=gootput_config_dict,
        )
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", type=str, default="http://localhost:8080/")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=100,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="xiaozheyao/MegaChat:sharegpt",
        help="Dataset to use for prompts.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=".local/benchmark_output",
        help="Output file to save the benchmark results.",
    )
    parser.add_argument(
        "--percentile-metrics",
        type=str,
        default="ttft,tpot,itl",
        help="Comma-seperated list of selected metrics to report percentils. "
        "This argument specifies the metrics to report percentiles. "
        'Allowed metric names are "ttft", "tpot", "itl", "e2el". '
        'Default value is "ttft,tpot,itl".',
    )
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="99",
        help="Comma-seperated list of percentiles for selected metrics. "
        'To report 25-th, 50-th, and 75-th percentiles, use "25,50,75". '
        'Default value is "99". '
        'Use "--percentile-metrics" to select metrics.',
    )
    parser.add_argument(
        "--goodput",
        nargs="+",
        required=False,
        help='Specify service level objectives for goodput as "KEY:VALUE" '
        "pairs, where the key is a metric name, and the value is in "
        'milliseconds. Multiple "KEY:VALUE" pairs can be provided, '
        "separated by spaces. Allowed request level metric names are "
        '"ttft", "tpot", "e2el". For more context on the definition of '
        "goodput, refer to DistServe paper: https://arxiv.org/pdf/2401.09670 "
        "and the blog: https://hao-ai-lab.github.io/blogs/distserve",
    )
    parser.add_argument(
        "--wait-until-ready",
        action="store_true",
        help="Wait until the server is ready before starting the benchmark.",
        default=True,
    )
    args = parser.parse_args()
    benchmark(args)
