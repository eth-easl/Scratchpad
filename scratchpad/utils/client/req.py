import os
from typing import Optional
import requests
from tenacity import retry, stop_after_attempt, wait_fixed


class LLM:
    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        if not base_url:
            base_url = os.environ.get("RC_API_BASE", None)
        if not api_key:
            api_key = os.environ.get("RC_API_KEY", None)
        if not base_url or not api_key:
            raise ValueError("API key or base_url not found")
        if not system_prompt:
            system_prompt = "You are a helpful assistant."
        self.model = model
        self.base_url = base_url + "/chat/completions"
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def __call__(self, prompt: str):
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        }
        try:
            res = requests.post(
                self.base_url,
                headers=self.headers,
                json=data,
            )
            result = res.json()
        except Exception as e:
            print(f"Error calling LLM: {res.text}")
            return None
        return result["choices"][0]["message"]["content"]
