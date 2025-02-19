import os
import json
import openai
import unittest
import subprocess
from pydantic import BaseModel, Field
from scratchpad.utils import kill_all_scratchpad_processes, wait_until_ready


class CapitalInfo(BaseModel):
    name: str = Field(..., description="Name of the capital city")


class TestScratchpadServer(unittest.TestCase):
    def setUp(self):
        kill_all_scratchpad_processes()
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = ".local"
        startup_cmd = "sp serve meta-llama/Llama-3.2-3B-Instruct --host 0.0.0.0 --port 8083 --grammar-backend xgrammar"
        self.proc = subprocess.Popen(startup_cmd, shell=True)
        self.client = openai.Client(base_url="http://127.0.0.1:8083/v1", api_key="None")
        wait_until_ready(host="127.0.0.1", port="8083")

    def tearDown(self):
        kill_all_scratchpad_processes()

    def test_context_length(self):
        # Test with long context
        long_message = "A " * 100  # Create a long input
        messages = [{"role": "user", "content": long_message}]
        try:
            response = self.client.chat.completions.create(
                model="meta-llama/Llama-3.2-3B-Instruct",
                messages=messages,
                max_tokens=2048,
            )

            assert (
                response.choices[0].finish_reason != "length"
            ), f"Expect finish_reason to be not `length`, got {response.choices[0].finish_reason}. Full response: {response}"
        except Exception as e:
            assert "context length exceeded" in str(e).lower()

        short_message = "Hello, how are you?"
        messages = [{"role": "user", "content": short_message}]

        response = self.client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct", messages=messages, max_tokens=100
        )
        assert (
            response.choices[0].finish_reason == "stop"
        ), f"Expected `stop`, got {response.choices[0].finish_reason}"

    def test_function_calling(self):
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
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
        messages = [
            {"role": "user", "content": "What's the weather like in Boston today?"}
        ]
        response = self.client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=messages,
            temperature=0.8,
            top_p=0.8,
            stream=False,
            tools=tools,
        )
        # verify response
        assert response.choices[0].message.content is not None
        try:
            result = json.loads(response.choices[0].message.content)
            assert (
                result["name"] == "get_current_weather"
            ), f"Expect the returned function name to be `get_current_weather`, got {result['name']}"
        except Exception as e:
            assert (
                False
            ), f"Expected response to be a valid JSON, got {response.choices[0].message.content}"

    def test_temperature_sampling(self):
        messages = [
            {"role": "user", "content": "Generate a random number between 1 and 10"}
        ]

        # Test with different temperature values
        responses = []
        for _ in range(5):
            response = self.client.chat.completions.create(
                model="meta-llama/Llama-3.2-3B-Instruct",
                messages=messages,
                temperature=1.0,  # High temperature for more randomness
                max_tokens=10,
            )
            responses.append(response.choices[0].message.content)

        # Verify that we get different responses
        assert (
            len(set(responses)) > 1
        ), "Temperature sampling not producing varied results"

    def test_structured_output(self):
        response = self.client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": "Where is capital of Switzerland?",
                },
            ],
            temperature=0.1,
            max_tokens=256,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "capital",
                    "schema": CapitalInfo.model_json_schema(),
                },
            },
        )
        try:
            response_content = response.choices[0].message.content
            response = response_content.strip()
            result = json.loads(response)
        except Exception as e:
            print(f"response: {response}")
            assert (
                False
            ), f"Expected response to be a valid JSON, got {response}"
        try:
            capital_info = CapitalInfo.model_validate_json(response)
        except Exception as e:
            assert False, f"Expected response to be a valid CapitalInfo, got {response}"

if __name__ == "__main__":
    unittest.main()
