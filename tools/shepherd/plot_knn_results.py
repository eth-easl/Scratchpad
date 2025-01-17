import json
import pandas as pd
from transformers import AutoTokenizer

from tools.shepherd.utils import (
    calculate_llms_accuracy,
    answer_mapping,
    pricings,
    build_prompt,
)

df = pd.read_csv(".local/shepherd/router_results.csv")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

with open(".local/shepherd/llm_responses_test.jsonl", "r") as fp:
    data = [json.loads(line) for line in fp]


def format_row_as_messages(row):
    row = row[1]
    system_prompt = f"The following are multiple choice questions (with answers) about {row['subject']}. Answer the question with one of the choices A, B, C, or D only, without any additional information or explanation."
    datum = [
        x
        for x in data
        if x["subject"] == row["subject"] and x["question"] == row["question"]
    ]
    if len(datum) == 0:
        raise ValueError(
            f"Could not find question {row['question']} for subject {row['subject']}"
        )
    datum = datum[0]
    row["choices"] = datum["choices"]
    user_prompt = build_prompt(row)["prompt"]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return messages


results = []

for row in df.iterrows():
    msg = format_row_as_messages(row)
    # count how many tokens we need
    tokens = tokenizer.apply_chat_template(msg, tokenize=True)
    res = {
        "token": len(tokens),
        "subject": row[1]["subject"],
        "answer": row[1]["answer"],
    }
    for model in pricings.keys():
        res[model] = row[1][model]
    res["router_output"] = row[1]["router_response"]
    res["router_model"] = row[1]["router_selected_model"]
    results.append(res)

df = pd.DataFrame(results)
aggregated = {}
for model in pricings.keys():
    df[f"{model}_correct"] = df[model] == df["answer"]
    df[f"{model}_correct"] = df[f"{model}_correct"].astype(int)
    accuracy = df[f"{model}_correct"].mean()
    aggregated[model] = {
        "accuracy": accuracy,
        "cost": df["token"].sum() * pricings[model],
    }

df["router_correct"] = df["router_output"] == df["answer"]
router_pricing = 0
for row in df.iterrows():
    row = row[1]
    router_used_model = row["router_model"]
    router_pricing += pricings[router_used_model] * row["token"]
aggregated["router"] = {"accuracy": df["router_correct"].mean(), "cost": router_pricing}

print(aggregated)
