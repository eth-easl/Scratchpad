import os
import openai

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
messages = [{"role": "user", "content": "What's the weather in Boston today?"}]

client = openai.Client(
    base_url=os.environ.get(f"RC_API_BASE"), api_key=os.environ.get(f"RC_API_KEY")
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-70B-Instruct",
    messages=messages,
    temperature=0.8,
    top_p=0.8,
    stream=False,
    tools=tools,
)
print(response)
