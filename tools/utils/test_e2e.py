import requests
import base64
import json
from openai import OpenAI

json_schema = json.dumps(
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "pattern": "^[\\w]+$"},
            "population": {"type": "integer"},
        },
        "required": ["name", "population"],
    }
)
prompt = "What is in this image? Return a JSON object"
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


def end_to_end(args):
    if args.multi_modal:
        base64_str = image_url_to_base64(img_url)
        messages = [
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
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

    client = OpenAI(
        base_url="http://localhost:8080/v1",
        api_key="test",
    )
    if args.constrained:
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "foo", "schema": json.loads(json_schema)},
        }
    else:
        response_format = None

    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
        response_format=response_format,
        stream=args.stream,
    )
    if args.stream:
        for chunk in response:
            if len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
    else:
        print(response.choices[0].message.content)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.2-11B-Vision-Instruct"
    )
    parser.add_argument("--stream", action="store_true", default=False)
    parser.add_argument("--constrained", action="store_true", default=False)
    parser.add_argument("--multi-modal", action="store_true", default=False)
    args = parser.parse_args()
    end_to_end(args)
