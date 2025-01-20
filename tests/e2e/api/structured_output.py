import os
import openai
import subprocess
from scratchpad.utils.client import wait_until_ready
from pydantic import BaseModel, Field


class CapitalInfo(BaseModel):
    name: str = Field(..., pattern=r"^\w+$", description="Name of the capital city")
    population: int = Field(..., description="Population of the capital city")


os.environ["PROMETHEUS_MULTIPROC_DIR"] = ".local"

startup_cmd = "sp serve meta-llama/Llama-3.2-3B-Instruct --host 0.0.0.0 --port 8080 --grammar-backend xgrammar"

proc = subprocess.Popen(startup_cmd, shell=True)

client = openai.Client(base_url="http://127.0.0.1:8080/v1", api_key="None")

wait_until_ready(host="127.0.0.1", port="8080")

response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Where is the capital of Switzerland?",
        },
    ],
    temperature=0,
    max_tokens=128,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "foo",
            # convert the pydantic model to json schema
            "schema": CapitalInfo.model_json_schema(),
        },
    },
)
response_content = response.choices[0].message.content
# validate the JSON response by the pydantic model
capital_info = CapitalInfo.model_validate_json(response_content)
print(f"Validated response: {capital_info.model_dump_json()}")
proc.communicate()
