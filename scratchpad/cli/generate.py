import requests

payload = {
    "text": "Once upon a time",
}

res = requests.post("http://localhost:3000/generate", json=payload)
print(res.json())
