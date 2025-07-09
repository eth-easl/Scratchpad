import os
import openai

client = openai.Client(api_key="test", base_url="http://localhost:8081/v1")
res = client.chat.completions.create(
    model="meta-llama/Llama-3.2-1B-Instruct",
    messages=[
        {
            "content": "Who is Pablo Picasso?",
            "role": "user",
        }
    ],
    stream=True,
    max_tokens=12,
    logprobs=True,
)
for chunk in res:
    print(chunk)
