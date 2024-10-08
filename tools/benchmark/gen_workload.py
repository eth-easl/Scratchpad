import os
import json
import argparse
import datasets
import tiktoken
import numpy as np
from loguru import logger
from arrival import PoissonProcess
from typing import Union

enc = tiktoken.get_encoding("cl100k_base")


def format_lmsys(prompt):
    return f"USER: {prompt}\nASSISTANT:"


def generate_model_distribution(distribution, num_queries, to_eval_models):
    if distribution == "uniform":
        return np.random.choice(to_eval_models, num_queries)
    if distribution == "distinct":
        return to_eval_models
    if distribution.startswith("zipf"):
        alpha = float(distribution.split(":")[1])
        assert alpha > 1, "alpha must be greater than 1"
        probs = [i**alpha for i in range(1, len(to_eval_models) + 1)]
        probs = np.array(probs) / sum(probs)
        return np.random.choice(to_eval_models, num_queries, p=probs)
    if distribution == "azure":
        import pandas as pd

        print("Using azure trace, loading...")
        df = pd.read_csv(
            "https://huggingface.co/datasets/xzyao/traces/resolve/main/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"
        )
        models = {}
        df["tstamp"] = df["end_timestamp"] - df["duration"]
        for row_id, row in df.iterrows():
            if row["func"] not in models:
                models[row["func"]] = 1
            else:
                models[row["func"]] += 1
        models = sorted(models.items(), key=lambda x: x[1], reverse=True)
        # take the first len(to_eval_models) models
        models = models[: len(to_eval_models)]
        # convert to probabilities
        total_calls = sum([x[1] for x in models])
        probs = [x[1] / total_calls for x in models]
        return np.random.choice(to_eval_models, num_queries, p=probs)
    raise ValueError("Invalid distribution")


def get_dialogs():
    trace = datasets.load_dataset("lmsys/chatbot_arena_conversations")["train"]
    all_dialogs = []
    response_tokens = []
    for idx, item in enumerate(trace):
        all_dialogs.append(format_lmsys(item["conversation_a"][0]["content"]))
        response_tokens.append(len(enc.encode(item["conversation_a"][1]["content"])))
    return all_dialogs, response_tokens


def generate_synthetic(args):
    print(args)
    to_eval_models = [
        "base-model",
    ]
    for i in range(1, args.num_models + 1, 1):
        to_eval_models.append(f"delta-{i}")
    print("Models to evaluate:", to_eval_models)

    poisson_ticks = PoissonProcess(args.arrival_rate).generate_arrivals(
        start=0, duration=args.duration
    )
    logger.info(f"Using Poisson arrival process, total_requests={len(poisson_ticks)}")

    traces_data = []
    dialogs, response_tokens = get_dialogs()
    models = generate_model_distribution(
        args.distribution,
        len(poisson_ticks),
        to_eval_models,
    )
    if args.distribution == "distinct":
        for idx in range(len(models)):
            traces_data.append(
                {
                    "id": idx,
                    "prompt": dialogs[idx],
                    "timestamp": poisson_ticks[idx],
                    "model": models[idx],
                    "min_tokens": (
                        args.gen_tokens
                        if args.gen_tokens != "auto"
                        else response_tokens[idx]
                    ),
                    "max_tokens": (
                        args.gen_tokens
                        if args.gen_tokens != "auto"
                        else response_tokens[idx]
                    ),
                }
            )
    else:
        for idx in range(len(poisson_ticks)):
            traces_data.append(
                {
                    "id": idx,
                    "prompt": dialogs[idx],
                    "timestamp": poisson_ticks[idx],
                    "model": models[idx],
                    "min_tokens": (
                        args.gen_tokens
                        if args.gen_tokens != "auto"
                        else response_tokens[idx]
                    ),
                    "max_tokens": (
                        args.gen_tokens
                        if args.gen_tokens != "auto"
                        else response_tokens[idx]
                    ),
                }
            )
    output_file = os.path.join(
        args.output,
        f"models={args.num_models},distribution={args.distribution},ar={args.arrival_rate},duration={args.duration}.jsonl",
    )
    with open(output_file, "w") as fp:
        for datum in traces_data:
            json.dump(datum, fp)
            fp.write("\n")


def main(args):
    generate_synthetic(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--distribution", type=str, default="uniform")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--gen-tokens", default=512)
    parser.add_argument("--arrival-rate", type=float, default=0)
    parser.add_argument("--duration", type=float, default=0)
    parser.add_argument("--num-models", type=int, default=8)

    args = parser.parse_args()
    main(args)

"""
Examples:
python .artifact/benchmarks/tools/trace/synthetic_generator.py --distribution uniform --gen-tokens 512 --arrival-rate 3 --duration 30 --output .artifact/workloads/
"""
