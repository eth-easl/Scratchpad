from scratchpad.extensions.shepherd import Route, Router
from scratchpad.utils.client import LLM, LLMEncoder
import datasets
import json
from tqdm import tqdm

from tools.shepherd.utils import (
    create_route_from_knn_builder,
    build_prompt,
    load_test_set,
)

test_ds = load_test_set()

encoder = LLMEncoder(
    model="meta-llama/Llama-3.2-1B-Instruct",
    base_url="http://localhost:8080/v1",
    api_key="test",
)
print("Creating routes")

routes = create_route_from_knn_builder(".local/shepherd/llm_responses_train.jsonl")
router = Router(encoder, routes)

print(f"Router created")
results = []
# dataset = [row for row in dataset if row["subject"] == subject]

for row in tqdm(test_ds):
    subject = row["subject"]
    router.set_system_prompt(
        f"The following are multiple choice questions (with answers) about {subject}. Answer the question with one of the choices A, B, C, or D only, without any additional information or explanation."
    )
    user_prompt = build_prompt(row)["prompt"]
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
with open(".local/shepherd/router_responses.jsonl", "w") as f:
    for res in results:
        f.write(json.dumps(res) + "\n")
