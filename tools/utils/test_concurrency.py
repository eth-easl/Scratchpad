import asyncio
import numpy as np
from tools.client.req import make_requests


def main(args):
    prompts = [
        "Who is Alan Turing?",
        "What is the capital of France?",
        "What is the capital of Germany?",
        "Who is Albert Einstein?",
    ]
    models = [
        # "eltorio/Llama-3.2-3B-appreciation-1",
        # "eltorio/Llama-3.2-3B-appreciation-2",
        "deltazip/meta-llama.Llama-3.2-3B-Instruct.4b_2n4m_128bs-1",
        "deltazip/meta-llama.Llama-3.2-3B-Instruct.4b_2n4m_128bs-2",
    ]
    prompts = np.random.choice(prompts, args.num_req, replace=True)
    models = np.random.choice(models, args.num_req, replace=True)
    print(models)
    reqs = [
        {
            "messages": [{"role": "user", "content": prompt}],
            "model": model,
            "max_tokens": 32,
        }
        for prompt, model in zip(prompts, models)
    ]
    responses = asyncio.run(make_requests(args.endpoint, reqs))
    for resp in responses:
        print(resp["choices"][0]["message"]["content"])


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
