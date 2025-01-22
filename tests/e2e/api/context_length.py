import os
import openai
import subprocess
from scratchpad.utils.client import wait_until_ready

os.environ["PROMETHEUS_MULTIPROC_DIR"] = ".local"

def test_context_length():
    startup_cmd = "sp serve meta-llama/Llama-3.2-3B-Instruct --host 0.0.0.0 --port 8083 --grammar-backend xgrammar"
    proc = subprocess.Popen(startup_cmd, shell=True)

    client = openai.Client(base_url="http://127.0.0.1:8083/v1", api_key="None")
    wait_until_ready(host="127.0.0.1", port="8083")

    # Test with long context
    long_message = "A " * 1000  # Create a long input
    messages = [{
        "role": "user",
        "content": long_message
    }]

    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=messages,
            max_tokens=100
        )
        assert response.choices[0].finish_reason != "length"
    except Exception as e:
        assert "context length exceeded" in str(e).lower()

    # Test with short context
    short_message = "Hello, how are you?"
    messages = [{
        "role": "user",
        "content": short_message
    }]

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=messages,
        max_tokens=100
    )
    assert response.choices[0].finish_reason == "stop"

    proc.terminate()
    proc.wait()

if __name__ == "__main__":
    test_context_length()