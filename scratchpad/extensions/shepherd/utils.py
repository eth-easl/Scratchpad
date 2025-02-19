import json
import random
from typing import List, Dict, Callable

from .route import Route
from scratchpad.utils import logger


def create_route_from_results(
    jsonl_path: str,
    answer_mapping: List,
    model_mapping: Dict,
    build_prompt: Callable,
    downsample_factor=1,
    cascade=True,
) -> List[Route]:
    with open(jsonl_path, "r") as f:
        data = [json.loads(line) for line in f]
    models = data[0]["output"].keys()
    routes = []
    remaining_data = [(idx, row) for idx, row in enumerate(data)]

    for model in models:
        # find correct answers for each model
        correct_utts = [
            (idx, row)
            for idx, row in enumerate(data)
            if row["output"][model] == answer_mapping[row["answer"]]
        ]
        remaining_data = [
            row for row in remaining_data if row[0] not in [x[0] for x in correct_utts]
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
                model_preferences=[model_mapping[model]],
            )
        )
    if len(remaining_data) > 0:
        logger.info(f"Remaining data: {len(remaining_data)}")
        remaining_data = [
            (row[0], build_prompt(row[1])["prompt"]) for row in remaining_data
        ]
        routes[0].add_utterances(remaining_data)
    return routes
