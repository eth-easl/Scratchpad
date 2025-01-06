import pandas as pd
import json


def load_jsonl(file_path):
    with open(file_path, "r") as file:
        data = [json.loads(line) for line in file]
    return pd.DataFrame(data)


def main(args):
    print(args)

    df = load_jsonl(args.data)
    print(df.head())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=".local/shepherd/scoring.jsonl")

    args = parser.parse_args()
    main(args)
