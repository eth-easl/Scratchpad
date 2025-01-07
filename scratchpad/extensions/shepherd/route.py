import json
from typing import List
from scratchpad.utils.client import LLM


class Route:
    def __init__(self, name: str, utterances: List[str], model_preferences: List):
        self.name = name
        self.utterances = utterances
        self.model_preferences = model_preferences


answer_mapping = ["A", "B", "C", "D"]

model_mappings = {
    "meta-llama/Llama-3.2-1B-Instruct": LLM(
        model="meta-llama/Llama-3.2-1B-Instruct",
        base_url="http://localhost:8080/v1",
        api_key="test",
    ),
    "meta-llama/Llama-3.2-3B-Instruct": LLM(
        model="meta-llama/Llama-3.2-3B-Instruct",
        base_url="http://localhost:8082/v1",
        api_key="test",
    ),
    "meta-llama/Llama-3.1-8B-Instruct": LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        base_url="http://localhost:8081/v1",
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
        print(f"Correct utterances for model {model}: {len(correct_utts)}")
        choices = ["A", "B", "C", "D"]
        reprompt = [
            f"{row['question']}\n"
            + "\n".join(
                [
                    f"{choice}. {answer}"
                    for choice, answer in zip(choices, row["choices"])
                ]
            )
            + "\nAnswer:"
            for row in correct_utts
        ]
        routes.append(
            Route(
                name=model,
                utterances=reprompt,
                model_preferences=[model_mappings[model]],
            )
        )
    return routes
