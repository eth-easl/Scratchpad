import os
import openai
import subprocess
from scratchpad.utils.client import wait_until_ready

os.environ["PROMETHEUS_MULTIPROC_DIR"] = ".local"

messages = [{
    "role": "user", 
    "content": "Write a short story about a robot learning to paint."
}]

def test_streaming_response():
    startup_cmd = "sp serve meta-llama/Llama-3.2-3B-Instruct --host 0.0.0.0 --port 8081 --grammar-backend xgrammar"
    proc = subprocess.Popen(startup_cmd, shell=True)

    client = openai.Client(base_url="http://127.0.0.1:8081/v1", api_key="None")
    wait_until_ready(host="127.0.0.1", port="8081")

    # Test streaming response
    stream = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=messages,
        temperature=0.7,
        stream=True
    )

    collected_messages = []
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            collected_messages.append(chunk.choices[0].delta.content)

    complete_response = ''.join(collected_messages)
    assert len(complete_response) > 0
    
    proc.terminate()
    proc.wait()

if __name__ == "__main__":
    test_streaming_response()