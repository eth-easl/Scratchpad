import json
from tools.shepherd.prompt_builder import get_response_models, available_models
from scratchpad.utils.client import LLM
from tqdm import tqdm
import multiprocessing as mp


def cleanup(datum):
    model = datum["model"]
    question = datum["question"]
    response = datum["response"]
    if response is None:
        llm = LLM(model=model, system_prompt="You are a helpful assistant.")
        response = llm(question)
        datum["response"] = response
    return datum


def main(args):
    print(args)
    models = ["meta-llama/Llama-3.1-8B-Instruct"]
    with open(args.input, "r") as fp:
        data = [json.loads(line) for line in fp]
    # responses = get_response_models(models, data)
    # with open(args.output, "w") as fp:
    #     for response in responses:
    #         fp.write(json.dumps(response) + "\n")
    empty_responses = [datum for datum in data if datum["response"] is None]
    print(f"len(empty_responses): {len(empty_responses)}")
    for i, datum in tqdm(enumerate(data)):
        if datum["response"] is None:
            llm = LLM(
                model=datum["model"], system_prompt="You are a helpful assistant."
            )
            data[i]["response"] = llm(datum["question"])
    with open(args.output, "w") as fp:
        for datum in data:
            fp.write(json.dumps(datum) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input", type=str, default=".data/calibration.jsonl")
    parser.add_argument("--output", type=str, default=".data/output.jsonl")
    main(parser.parse_args())
