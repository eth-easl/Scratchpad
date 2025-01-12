from scratchpad.extensions.shepherd import Route, Router
from scratchpad.utils.client import LLM, LLMEncoder
import datasets
import json
from tqdm import tqdm

from tools.shepherd.utils import create_route_from_knn_builder

dataset = datasets.load_dataset("cais/mmlu", "all")["test"]

encoder = LLMEncoder(
    model="meta-llama/Llama-3.2-1B-Instruct",
    base_url="http://localhost:8083/v1",
    api_key="test",
)

routes = create_route_from_knn_builder(".local/shepherd/knn_builder.jsonl")
router = Router(encoder, routes)

choices = ["A", "B", "C", "D"]
results = []
subjects = set([row["subject"] for row in dataset])

subject = subjects.pop()
# dataset = [row for row in dataset if row["subject"] == subject]

for row in tqdm(dataset):
    subject = row["subject"]
    router.set_system_prompt(
        f"The following are multiple choice questions (with answers) about {subject}. Answer the question with one of the choices A, B, C, or D only, without any additional information or explanation."
    )
    user_prompt = (
        f"{row['question']}\n"
        + "\n".join(
            [f"{choice}. {answer}" for choice, answer in zip(choices, row["choices"])]
        )
        + "\nAnswer:"
    )
    selected_model, response = router(user_prompt, max_tokens=1, temperature=0.001)
    results.append(
        {
            "subject": subject,
            "question": row["question"],
            "choices": row["choices"],
            "answer": row["answer"],
            "selected_model": selected_model,
            "output": response,
        }
    )
# with open(".local/shepherd/router_response_2.jsonl", "w") as f:
#     for res in results:
#         f.write(json.dumps(res) + "\n")
