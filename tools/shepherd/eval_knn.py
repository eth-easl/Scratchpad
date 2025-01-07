from scratchpad.extensions.shepherd import Route, Router, create_route_from_knn_builder
from scratchpad.utils.client import LLM, LLMEncoder
import datasets
import json
from tqdm import tqdm

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
    with open(f".local/shepherd/router_response.jsonl", "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
