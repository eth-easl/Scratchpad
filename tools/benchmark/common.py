# adapted from https://github.com/vllm-project/vllm/blob/main/benchmarks/backend_request_func.py
import json
import os
import asyncio
import numpy as np
import sys
import aiohttp
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Union, Tuple, AsyncGenerator
from tqdm.asyncio import tqdm
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from tqdm import tqdm

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    best_of: int = 1
    logprobs: Optional[int] = None
    multi_modal_content: Optional[dict] = None
    ignore_eos: bool = False


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    error: str = ""


async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        ("completions", "profile")
    ), "OpenAI Completions API URL must end with 'completions' or 'profile'."

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "model": request_func_input.model,
            "prompt": request_func_input.prompt,
            "temperature": 0.0,
            "best_of": request_func_input.best_of,
            "max_tokens": request_func_input.output_len,
            "logprobs": request_func_input.logprobs,
            "stream": True,
            "ignore_eos": request_func_input.ignore_eos,
        }
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                        if chunk == "[DONE]":
                            latency = time.perf_counter() - st
                        else:
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if data["choices"][0]["text"]:
                                timestamp = time.perf_counter()
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += data["choices"][0]["text"]

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def construct_dataset(
    endpoint: str, dataset_id: str, tokenizer_id: str, size: int = -1
) -> List[RequestFuncInput]:
    dataset_name, dataset_split = dataset_id.split(":")
    print(f"Constructing dataset {dataset_name}, split: {dataset_split}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    dataset = load_dataset(dataset_name, dataset_split)
    if not isinstance(dataset, DatasetDict):
        raise ValueError(
            f"Dataset {dataset_id} not supported, got type {type(dataset)}"
        )
    dataset = dataset["train"]
    if size != -1:
        dataset = dataset.select(range(size))
    requests: List[RequestFuncInput] = []
    for req in tqdm(dataset):
        conversations = req["conversations"]
        len_convs = len(conversations) // 2
        for i in range(len_convs):
            prompt = conversations[2 * i]["content"]
            response = conversations[2 * i + 1]["content"]
            req = RequestFuncInput(
                prompt=prompt,
                api_url=endpoint,
                prompt_len=len(tokenizer(prompt)["input_ids"]),
                output_len=len(tokenizer(response)["input_ids"]),
                model=tokenizer_id,
                ignore_eos=True,
            )
            requests.append(req)
    return requests
