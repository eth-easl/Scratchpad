import json
import torch
import random
import datasets
from typing import List

from scratchpad.utils.client import LLM
from scratchpad.extensions.shepherd import Route

answer_mapping = ["A", "B", "C", "D"]

model_mappings = {
    "meta-llama/Llama-3.2-1B-Instruct": LLM(
        model="meta-llama/Llama-3.2-1B-Instruct",
        base_url="http://localhost:8081/v1",
        api_key="test",
    ),
    "meta-llama/Llama-3.2-3B-Instruct": LLM(
        model="meta-llama/Llama-3.2-3B-Instruct",
        base_url="http://localhost:8082/v1",
        api_key="test",
    ),
    "meta-llama/Llama-3.1-8B-Instruct": LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        base_url="http://localhost:8083/v1",
        api_key="test",
    ),
    "meta-llama/Llama-3.3-70B-Instruct": LLM(
        model="meta-llama/Llama-3.3-70B-Instruct",
    ),
}
# cheapest pricing from openrouter.ai for bf16, per million tokens
pricings = {
    "meta-llama/Llama-3.2-1B-Instruct": 0.01,
    "meta-llama/Llama-3.2-3B-Instruct": 0.015,
    "meta-llama/Llama-3.1-8B-Instruct": 0.025,
    "meta-llama/Llama-3.3-70B-Instruct": 0.39,
}
penalty = torch.tensor([pricings[model] for model in model_mappings.keys()])
penalty = penalty / penalty.sum()


def create_route_from_knn_builder(
    jsonl_path: str, downsample_factor=1, cascade=True
) -> List[Route]:
    with open(jsonl_path, "r") as f:
        data = [json.loads(line) for line in f]
    models = data[0]["output"].keys()
    routes = []
    for model in models:
        # find correct answers for each model
        correct_utts = [
            (idx, row)
            for idx, row in enumerate(data)
            if row["output"][model] == answer_mapping[row["answer"]]
        ]
        if cascade:
            data = [row for row in data if row not in correct_utts]
        correct_utts = random.sample(
            correct_utts, len(correct_utts) // downsample_factor
        )
        reprompt = [(row[0], build_prompt(row[1])["prompt"]) for row in correct_utts]
        routes.append(
            Route(
                name=model,
                utterances=reprompt,
                model_preferences=[model_mappings[model]],
            )
        )
    return routes


def calculate_llms_accuracy(data):
    models = data[0]["output"].keys()
    accuracies = {}
    for model in models:
        correct = 0
        for row in data:
            if answer_mapping[row["answer"]] == row["output"][model]:
                correct += 1
        accuracies[model] = correct / len(data)
    return accuracies


def build_prompt(row):
    """ """
    user_prompt = (
        f"{row['question']}\n"
        + "\n".join(
            [
                f"{choice}. {answer}"
                for choice, answer in zip(answer_mapping, row["choices"])
            ]
        )
        + "\nAnswer:"
    )
    row["prompt"] = user_prompt
    return row


def construct_ds(test_ratio: float = 0.2, seed=42):
    random.seed(seed)
    dataset = datasets.load_dataset("cais/mmlu", "all")["test"]
    dataset = dataset.shuffle(seed=seed)
    results = dataset.train_test_split(test_size=test_ratio)

    train, test = results["train"], results["test"]
    train = train.map(lambda x: build_prompt(x))
    test = test.map(lambda x: build_prompt(x))
    return train, test


def load_test_set():
    with open(".local/shepherd/llm_responses_test.jsonl", "r") as f:
        data = [json.loads(line) for line in f]
    return data


def create_trainset_from_jsonl(filename: str):
    with open(filename, "r") as f:
        data = [json.loads(line) for line in f]
