import os
import json
from tqdm import tqdm
import pandas as pd

from scratchpad.extensions.shepherd import Route, Router
from scratchpad.utils.client import LLM, LLMEncoder
from tools.shepherd.utils import (
    create_route_from_knn_builder,
    build_prompt,
    load_test_set,
    answer_mapping,
    pricings,
)

os.environ["OMP_NUM_THREADS"] = "16"

test_ds = load_test_set()

encoder = LLMEncoder(
    model="meta-llama/Llama-3.2-1B-Instruct",
    base_url="http://localhost:8080/v1",
    api_key="test",
)
routes = create_route_from_knn_builder(
    ".local/shepherd/llm_responses_train.jsonl",
    downsample_factor=1,
    cascade=False,
)

with open(".local/shepherd/llm_responses_test.jsonl") as f:
    data = [json.loads(line) for line in f]

router = Router(
    encoder,
    routes,
    index_location=".local/shepherd",
    policy="learned",
    cost=pricings,
)
router_data = []
results = []

for row in tqdm(test_ds):
    user_prompt = build_prompt(row)["prompt"]
    selected_model, response = router(
        user_prompt,
        max_tokens=1,
        temperature=0.001,
        dry_run=True,
        verbose=False,
        k=5,
    )
    output = row["output"][selected_model]
    router_data.append(
        {
            "subject": row["subject"],
            "question": row["question"],
            "choices": row["choices"],
            "answer": row["answer"],
            "selected_model": selected_model,
            "output": output,
        }
    )
print(f"Stats: {router.stats}")

for datum in data:
    res = {
        "subject": datum["subject"],
        "question": datum["question"],
        "answer": answer_mapping[datum["answer"]],
    }
    for model in datum["output"]:
        res[model] = datum["output"][model]
    router_datum = [
        x
        for x in router_data
        if x["question"] == datum["question"]
        and x["choices"] == datum["choices"]
        and x["subject"] == datum["subject"]
    ]
    if len(router_datum) > 0:
        router_datum = router_datum[0]
        res["router_selected_model"] = router_datum["selected_model"]
        res["router_response"] = router_datum["output"]
    results.append(res)

df = pd.DataFrame(results)
df.to_csv(".local/shepherd/router_results.csv", index=False)

models = list(pricings.keys()) + ["router_response"]
for model in models:
    df[f"{model}_correct"] = df[model] == df["answer"]
    df[f"{model}_correct"] = df[f"{model}_correct"].astype(int)
    print(f"Model: {model}, Accuracy: {df[f'{model}_correct'].mean()}")
