import requests

for i in range(100):

    res = requests.post(
        "http://localhost:8080/v1/completions",
        json={
            "model": "meta-llama/Llama-3.2-1B-Instruct",
            "prompt": "Once upon a time",
            "max_tokens": 128,
        },
    )
