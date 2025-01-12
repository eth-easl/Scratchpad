import os
from typing import Optional, List
import requests
import numpy as np
from tenacity import retry, stop_after_attempt, wait_fixed


class Encoder:
    pass


class LLMEncoder(Encoder):
    def __init__(self, model: str, base_url: Optional[str], api_key: Optional[str]):
        super().__init__()
        if not base_url:
            base_url = os.environ.get("RC_API_BASE", None)
        if not api_key:
            api_key = os.environ.get("RC_API_KEY", None)
        if not base_url or not api_key:
            raise ValueError("API key or endpoint not found")
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def __call__(self, docs: List[str], truncate: bool = False) -> List[List[float]]:
        if truncate:
            raise NotImplementedError("Truncation not supported yet")
        data = {
            "model": self.model,
            "input": docs,
            "encoding_format": "float",
        }
        try:
            res = requests.post(
                self.base_url + "/embeddings",
                headers=self.headers,
                json=data,
            )
            result = res.json()["data"]
        except Exception as e:
            print(f"Error calling LLM: {res.text}")
            return None
        return np.array([(r["embedding"]) for r in result])


if __name__ == "__main__":
    encoder = LLMEncoder(
        model="meta-llama/Llama-3.2-1B-Instruct",
        base_url="http://localhost:8080/v1",
        api_key="test",
    )
    response = encoder(["Test", "Who is alan turing?"])
