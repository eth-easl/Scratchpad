import datasets
import json
from tqdm import tqdm
from scratchpad.utils.client import LLM

from tools.shepherd.utils import construct_ds

local_1b = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    base_url="http://localhost:8081/v1",
    api_key="test",
)
local_3b = LLM(
    model="meta-llama/Llama-3.2-3B-Instruct",
    base_url="http://localhost:8082/v1",
    api_key="test",
)
local_8b = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    base_url="http://localhost:8083/v1",
    api_key="test",
)
remote_70b = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
)
llms = [local_1b, local_3b, local_8b, remote_70b]


def prepare_data(dataset):
    results = []
    for row in tqdm(dataset):
        subject = row["subject"]
        llm_answers = {}
        for llm in llms:
            llm.set_system_prompt(
                f"The following are multiple choice questions (with answers) about {subject}. Answer the question with one of the choices A, B, C, or D only, without any additional information or explanation."
            )
            response = llm(row["prompt"], max_tokens=1, temperature=0.001)
            llm_answers[llm.model] = response
        results.append(
            {
                "subject": subject,
                "question": row["question"],
                "choices": row["choices"],
                "answer": row["answer"],
                "output": llm_answers,
            }
        )
    return results


def main(args):
    train, test = construct_ds(test_ratio=0.1, seed=42)
    with open(f".local/shepherd/llm_train_91.jsonl", "w") as f:
        for row in train:
            f.write(json.dumps(row) + "\n")
    with open(f".local/shepherd/llm_test_91.jsonl", "w") as f:
        for row in test:
            f.write(json.dumps(row) + "\n")

    train_results = prepare_data(train)
    test_results = prepare_data(test)

    with open(f".local/shepherd/llm_responses_test_91.jsonl", "w") as f:
        for result in test_results:
            f.write(json.dumps(result) + "\n")
    print(f"Finish writing test results to .local/shepherd/llm_responses_test.jsonl")

    with open(f".local/shepherd/llm_responses_train_91.jsonl", "w") as f:
        for result in train_results:
            f.write(json.dumps(result) + "\n")
    print(f"Finish writing train results to .local/shepherd/llm_responses_train.jsonl")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
