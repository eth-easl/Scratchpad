import json
from tools.shepherd.utils import decontamination, contain_contamination


def fix_contamination(train, test):
    new_train = decontamination(test, train)
    return new_train


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", type=str, default=".local/shepherd/llm_responses_train.jsonl"
    )
    parser.add_argument(
        "--test", type=str, default=".local/shepherd/llm_responses_test.jsonl"
    )
    parser.add_argument(
        "--new-train",
        type=str,
        default=".local/shepherd/llm_responses_train_fixed.jsonl",
    )

    args = parser.parse_args()
    with open(args.test) as f:
        test = [json.loads(line) for line in f]
    with open(args.train) as f:
        train = [json.loads(line) for line in f]
    train = fix_contamination(train, test)
    has_contamination = contain_contamination(test, train)
    if has_contamination:
        print("Found contamination")
    else:
        print("No contamination found")

    with open(args.new_train, "w") as f:
        for t in train:
            f.write(json.dumps(t) + "\n")
