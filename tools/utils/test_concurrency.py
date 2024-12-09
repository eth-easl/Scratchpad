from tools.client.req import make_requests
import numpy as np
import asyncio


def main(args):
    print(args)
    prompts = [
        "Who is Alan Turing?",
        "What is the capital of France?",
        "What is the capital of Germany?",
        "What is the capital of Spain?",
        "Who is the president of the United States?",
        "Who is the president of France?",
        "Who is Albert Einstein?",
    ]
    # randomly pick num_req prompts
    prompts = np.random.choice(prompts, args.num_req, replace=True)
    reqs = [
        {
            "messages": [{"role": "user", "content": prompt}],
            "model": args.model,
            "max_tokens": 32,
        }
        for prompt in prompts
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
