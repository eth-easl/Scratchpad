import requests

payload = {
    "text": "Alan Turing is",
    "sampling_params": {
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
    },
}

res = requests.post("http://localhost:3000/generate", json=payload)
print(res.json())
