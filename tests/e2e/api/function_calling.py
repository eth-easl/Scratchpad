import os
import openai
import subprocess
from scratchpad.utils.client import wait_until_ready
from pydantic import BaseModel, Field

os.environ["PROMETHEUS_MULTIPROC_DIR"] = ".local"

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]
messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]

startup_cmd = "sp serve meta-llama/Llama-3.2-3B-Instruct --host 0.0.0.0 --port 8080 --grammar-backend xgrammar"

proc = subprocess.Popen(startup_cmd, shell=True)

client = openai.Client(base_url="http://127.0.0.1:8080/v1", api_key="None")

wait_until_ready(host="127.0.0.1", port="8080")

response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    messages=messages,
    temperature=0.8,
    top_p=0.8,
    stream=False,
    tools=tools,
)
print(response)

proc.communicate()
