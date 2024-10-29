import time
import asyncio
import requests
import numpy as np
from typing import List, Tuple, AsyncGenerator, Optional, Callable, Dict
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from tools.benchmark.common import (
    construct_dataset,
    get_request,
    async_request_openai_completions,
    RequestFuncOutput,
    RequestFuncInput,
    calculate_metrics,
)
from tools.benchmark.report import print_benchmark


def check_goodput_args(args):
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
    input_requests: List[RequestFuncInput],
    request_rate: float,
    request_func: Callable,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentile_metrics: List[str],
    selected_percentiles: List[str],
    goodput_config_dict: Dict[str, float],
    max_concurrency: Optional[int] = None,
):
    pbar = tqdm(total=len(input_requests))
    tasks: List[asyncio.Task] = []
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            return await request_func(request_func_input=request_func_input, pbar=pbar)
        async with semaphore:
            return await request_func(request_func_input=request_func_input, pbar=pbar)

    benchmark_start_time = time.perf_counter()
    async for request in get_request(input_requests, request_rate):
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
        selected_percentiles=selected_percentiles,
        selected_percentile_metrics=selected_percentile_metrics,
        goodput_config_dict=goodput_config_dict,
    )
    print_benchmark(metrics)
    return metrics


def benchmark(args):
    print(args)
    if args.model == "":
        args.model = args.tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    requests = construct_dataset(
        args.endpoint, args.dataset, tokenizer, args.num_prompts
    )
    for req in requests:
        req.model = args.model
    request_func = async_request_openai_completions
    gootput_config_dict = check_goodput_args(args)
    asyncio.run(
        run_benchmark(
            requests,
            args.request_rate,
            request_func,
            tokenizer,
            args.percentile_metrics.split(","),
            selected_percentiles=[float(p) for p in args.metric_percentiles.split(",")],
            goodput_config_dict=gootput_config_dict,
        )
    )


if __name__ == "__main__":
    import argparse

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
        default=1000,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="xiaozheyao/MegaChat:sharegpt",
        help="Dataset to use for prompts.",
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
    args = parser.parse_args()
    benchmark(args)
