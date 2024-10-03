import os


def benchmark_quality(
    model_name: str,
    url: str,
    tasks: str,
    num_fewshot: int = 0,
    instruct_model: bool = False,
):
    job = f"lm_eval --model local-completions --model_args model={model_name},base_url={url},tokenized_requests=False,num_concurrent=3,max_retries=5 --tasks {tasks} --num_fewshot {num_fewshot}"
    if instruct_model:
        job += " --apply_chat_template"
    print(f"Running benchmark with: {job}")
    os.system(job)
