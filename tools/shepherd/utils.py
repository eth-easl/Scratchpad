import json
from typing import List
import datasets
from scratchpad.utils.client import LLM
from scratchpad.extensions.shepherd import Route, Router
import random

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


def create_route_from_knn_builder(jsonl_path: str) -> List[Route]:
    with open(jsonl_path, "r") as f:
        data = [json.loads(line) for line in f]
    models = data[0]["output"].keys()
    routes = []
    for model in models:
        # find correct answers for each model
        correct_utts = [
            row for row in data if row["output"][model] == answer_mapping[row["answer"]]
        ]
        correct_utts = random.sample(correct_utts, 100)
        reprompt = [build_prompt(row)["prompt"] for row in correct_utts]

        # remove correct_utts from entire data
        data = [row for row in data if row not in correct_utts]
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
    with open(".local/shepherd/llm_responses_tests.jsonl", "r") as f:
        data = [json.loads(line) for line in f]
    return data
