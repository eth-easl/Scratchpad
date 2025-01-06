import datasets
from scratchpad.utils.client import LLM
from tqdm import tqdm
import json


local_1b = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    base_url="http://localhost:8080/v1",
    api_key="test",
)
local_3b = LLM(
    model="meta-llama/Llama-3.2-3B-Instruct",
    base_url="http://localhost:8082/v1",
    api_key="test",
)
local_8b = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    base_url="http://localhost:8081/v1",
    api_key="test",
)
remote_70b = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
)
llms = [local_1b, local_3b, local_8b, remote_70b]


def main(args):
    results = []
    dataset = datasets.load_dataset(args.dataset, "all")
    choices = ["A", "B", "C", "D"]
    for row in tqdm(dataset["dev"]):
        subject = row["subject"]
        llm_answers = {}
        for llm in llms:
            llm.set_system_prompt(
                f"The following are multiple choice questions (with answers) about {subject}. Answer the question with one of the choices A, B, C, or D only, without any additional information or explanation."
            )
            user_prompt = (
                f"{row['question']}\n"
                + "\n".join(
                    [
                        f"{choice}. {answer}"
                        for choice, answer in zip(choices, row["choices"])
                    ]
                )
                + "\nAnswer:"
            )
            response = llm(user_prompt, max_tokens=1, temperature=0.001)

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
        print(results)
    with open(f".local/shepherd/knn_builder.jsonl", "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cais/mmlu")
    args = parser.parse_args()
    main(args)
