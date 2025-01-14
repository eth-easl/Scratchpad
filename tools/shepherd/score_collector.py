import json
from tqdm import tqdm
from tools.shepherd.prompt_builder import judge_response


def score(args):
    with open(args.input, "r") as f:
        data = [json.loads(x) for x in f.readlines()]
    data = data[::-1]
    empty_data = [x for x in data if x["reason"] == "Error"]
    print(f"Empty data: {len(empty_data)}")
    for datum in tqdm(data):
        if datum["reason"] == "Error":
            datum["score"], datum["reason"] = judge_response(
                args.judge_model, datum["question"], datum["response"]
            )
            print(f"Score: {datum['score']}, Reason: {datum['reason']}")
    # for each model, print average score
    models = set([x["model"] for x in data])
    for model in models:
        model_data = [x for x in data if x["model"] == model]
        scores = [x["score"] for x in model_data]
        print(
            f"Model: {model}, Average score: {sum(scores) / len(scores)}, Low score: {min(scores)}, High score: {max(scores)}"
        )
    with open(args.output, "w") as f:
        for datum in data:
            f.write(json.dumps(datum) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scoring responses")
    parser.add_argument("--input", type=str, help="Input file")
    parser.add_argument(
        "--judge-model", type=str, default="meta-llama/Llama-3.3-70B-Instruct"
    )
    parser.add_argument("--output", type=str, default=".data/scoring.jsonl")
    score(parser.parse_args())
