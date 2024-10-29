import requests
from tools.benchmark.common import construct_dataset, RequestFuncInput


def benchmark(args):
    print(args)
    requests = construct_dataset(
        args.endpoint, args.dataset, args.tokenizer, args.num_prompts
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", type=str, default="http://localhost:8080/")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.2-1B")
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
