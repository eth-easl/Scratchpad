import json
import datasets
import asyncio
from tools.shepherd.prompt_builder import is_good_question


def prepare_dataset(args):
    print(args)
    dataset = datasets.load_dataset(args.dataset)
    dataset = dataset.shuffle(seed=args.seed)["train_sft"]
    dataset = dataset["prompt"]
    questions = []
    for datum in dataset:
        score = is_good_question(datum)
        if score > 4:
            questions.append({"question": datum, "score": score})
            print(
                f"[len(questions)/{args.num_samples}] Score: {score}, Question: {datum}"
            )
        if len(questions) >= args.num_samples:
            break
    questions = sorted(questions, key=lambda x: x["score"], reverse=True)
    with open(".local/calibration.jsonl", "w") as f:
        for question in questions:
            f.write(json.dumps(question) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare datasets")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=256)
    prepare_dataset(parser.parse_args())
