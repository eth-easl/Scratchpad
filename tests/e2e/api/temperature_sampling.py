import os
import openai
import subprocess
from scratchpad.utils.client import wait_until_ready

os.environ["PROMETHEUS_MULTIPROC_DIR"] = ".local"

messages = [{
    "role": "user", 
    "content": "Generate a random number between 1 and 10"
}]

def test_temperature_sampling():
    startup_cmd = "sp serve meta-llama/Llama-3.2-3B-Instruct --host 0.0.0.0 --port 8082 --grammar-backend xgrammar"
    proc = subprocess.Popen(startup_cmd, shell=True)

    client = openai.Client(base_url="http://127.0.0.1:8082/v1", api_key="None")
    wait_until_ready(host="127.0.0.1", port="8082")

    # Test with different temperature values
    responses = []
    for _ in range(5):
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=messages,
            temperature=1.0,  # High temperature for more randomness
            max_tokens=10
        )
        responses.append(response.choices[0].message.content)

    # Verify that we get different responses
    assert len(set(responses)) > 1, "Temperature sampling not producing varied results"

    # Test with temperature = 0
    deterministic_responses = []
    for _ in range(3):
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=messages,
            temperature=0.0,  # Zero temperature for deterministic output
            max_tokens=10
        )
        deterministic_responses.append(response.choices[0].message.content)

    # Verify that responses are identical with temperature = 0
    assert len(set(deterministic_responses)) == 1, "Deterministic sampling not consistent"

    proc.terminate()
    proc.wait()

if __name__ == "__main__":
    test_temperature_sampling()