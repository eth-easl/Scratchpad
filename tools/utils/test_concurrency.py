import asyncio
import numpy as np
from tools.client.req import make_requests


def main(args):
    print(args)
    prompts = [
        "Who is Alan Turing?",
        "What is the capital of France?",
    ]
    models = [
        "eltorio/Llama-3.2-3B-appreciation-1",
        "eltorio/Llama-3.2-3B-appreciation-1",
    ]
    prompts = prompts[: args.num_req]
    reqs = [
        {
            "messages": [{"role": "user", "content": prompt}],
            "model": model,
            "max_tokens": 32,
        }
        for prompt, model in zip(prompts, models)
    ]
    responses = asyncio.run(make_requests(args.endpoint, reqs))
    print(responses)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint", type=str, default="http://localhost:8080/v1/chat/completions"
    )
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.2-11B-Vision-Instruct"
    )
    parser.add_argument("--num-req", type=int, default=2)
    args = parser.parse_args()
    main(args)
