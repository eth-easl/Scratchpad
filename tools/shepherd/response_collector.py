import json
from tools.shepherd.prompt_builder import get_response_models, available_models
from scratchpad.utils.client import LLM
from tqdm import tqdm


def main(args):
    print(args)
    with open(args.input, "r") as fp:
        data = [json.loads(line) for line in fp]
    for datum in tqdm(data):
        model = datum["model"]
        question = datum["question"]
        response = datum["response"]
        if response is None:
            llm = LLM(model=model, system_prompt="You are a helpful assistant.")
            response = llm(question)
            datum["response"] = response

    with open(args.output, "w") as fp:
        for resp in data:
            fp.write(json.dumps(resp) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input", type=str, default=".data/calibration.jsonl")
    parser.add_argument("--output", type=str, default=".data/output.jsonl")
    main(parser.parse_args())
