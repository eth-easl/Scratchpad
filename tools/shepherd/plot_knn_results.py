import json
import pandas as pd
from tools.shepherd.utils import calculate_llms_accuracy, answer_mapping

# cheapest pricing from openrouter.ai for bf16, per million tokens
pricings = {
    "meta-llama/Llama-3.2-1B-Instruct": 0.01,
    "meta-llama/Llama-3.2-3B-Instruct": 0.015,
    "meta-llama/Llama-3.1-8B-Instruct": 0.025,
    "meta-llama/Llama-3.3-70B-Instruct": 0.39,
}

results = []

with open(".local/shepherd/llm_responses.jsonl") as f:
    data = [json.loads(line) for line in f]

with open(".local/shepherd/router_response_2.jsonl") as f:
    router_data = [json.loads(line) for line in f]


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
    # assert len(router_datum) == 1, f"More than one router response found: {router_datum}"
    router_datum = router_datum[0]
    res["router_selected_model"] = router_datum["selected_model"]
    res["router_response"] = router_datum["output"]

    results.append(res)

df = pd.DataFrame(results)
df.to_csv(".local/shepherd/llm_router_comparison.csv", index=False)

with open(".local/shepherd/llm_router_comparison.csv") as f:
    df = pd.read_csv(f)
    models = list(pricings.keys()) + ["router_response"]
    # calculate accuracy per model
    for model in models:
        df[f"{model}_correct"] = df[model] == df["answer"]
        df[f"{model}_correct"] = df[f"{model}_correct"].astype(int)
        print(f"Model: {model}, Accuracy: {df[f'{model}_correct'].mean()}")
