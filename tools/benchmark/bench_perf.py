import requests
import asyncio
import numpy as np
from typing import List, Tuple, AsyncGenerator
from tqdm.asyncio import tqdm
from typing import Optional, Callable
from tools.benchmark.common import (
    construct_dataset,
    get_request,
    async_request_openai_completions,
    RequestFuncOutput,
    RequestFuncInput,
)


async def run_benchmark(
    input_requests: List[RequestFuncInput],
    request_rate: float,
    request_func: Callable,
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

    async for request in get_request(input_requests, request_rate):
        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input=request, pbar=pbar)
            )
        )
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)


def benchmark(args):
    print(args)
    if args.model == "":
        args.model = args.tokenizer

    requests = construct_dataset(
        args.endpoint, args.dataset, args.tokenizer, args.num_prompts
    )
    for req in requests:
        req.model = args.model
    request_func = async_request_openai_completions
    asyncio.run(run_benchmark(requests, args.request_rate, request_func))


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
    args = parser.parse_args()
    benchmark(args)
