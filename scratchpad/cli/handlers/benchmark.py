import os
from typing import List


def benchmark_quality(model_name: str, url: str, tasks: str, num_fewshot: int = 0):
    job = f"lm_eval --model local-completions --model_args model={model_name},base_url={url},tokenized_requests=False,num_concurrent=10 --tasks {tasks} --num_fewshot {num_fewshot}"
    print(f"Running benchmark with: {job}")
    os.system(job)
