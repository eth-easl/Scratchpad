from scratchpad.extensions.shepherd import Route, Router
from scratchpad.utils.client import LLM, LLMEncoder
import datasets
import json
from tqdm import tqdm
import pandas as pd
from tools.shepherd.utils import (
    create_route_from_knn_builder,
    build_prompt,
    load_test_set,
    answer_mapping,
    pricings,
)

test_ds = load_test_set()

encoder = LLMEncoder(
    model="meta-llama/Llama-3.2-1B-Instruct",
    base_url="http://localhost:8080/v1",
    api_key="test",
)
routes = create_route_from_knn_builder(
    ".local/shepherd/llm_responses_train.jsonl", downsample_factor=1
)

with open(".local/shepherd/llm_responses_test.jsonl") as f:
    data = [json.loads(line) for line in f]

router = Router(encoder, routes, index_location=".local/shepherd")
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
        k=1,
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
    assert (
        len(router_datum) == 1
    ), f"More than one router response found: {router_datum}"

    if len(router_datum) > 0:
        router_datum = router_datum[0]
        res["router_selected_model"] = router_datum["selected_model"]
        res["router_response"] = router_datum["output"]

    results.append(res)

print(f"router: {router.stats}")

df = pd.DataFrame(results)
df.to_csv(".local/shepherd/router_results.csv", index=False)
print(df.head())
models = list(pricings.keys()) + ["router_response"]
for model in models:
    df[f"{model}_correct"] = df[model] == df["answer"]
    df[f"{model}_correct"] = df[f"{model}_correct"].astype(int)
    print(f"Model: {model}, Accuracy: {df[f'{model}_correct'].mean()}")
