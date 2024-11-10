import requests
import base64
from openai import OpenAI

prompt = "What is in this image?"
img_url = "https://images.unsplash.com/photo-1692350914621-f0ca2d206368?q=80&w=3000&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
stream = True


def image_url_to_base64(url):
    # Fetch the image from the URL
    response = requests.get(url)

    # Ensure the request was successful
    if response.status_code == 200:
        # Encode the image content in base64
        base64_str = base64.b64encode(response.content).decode("utf-8")
        return base64_str
    else:
        raise Exception(
            f"Failed to retrieve image from URL. Status code: {response.status_code}"
        )


base64_str = image_url_to_base64(img_url)
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="test",
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-11B-Vision-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:jpeg;base64,{base64_str}"},
                },
            ],
        }
    ],
    stream=stream,
)
if stream:
    for chunk in response:
        if len(chunk.choices) > 0 and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
else:
    print(response.choices[0].message.content)
