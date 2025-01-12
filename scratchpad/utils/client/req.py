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
        price_per_mtokens: Optional[float] = None,
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
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self.price_per_million_tokens = price_per_mtokens

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt

    def set_pricing_per_mtokens(self, price: float):
        self.price_per_million_tokens = price

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def __call__(self, prompt: str, max_tokens: int = 10, temperature: float = 0.5):
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
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
        self._prompt_tokens += result["usage"]["prompt_tokens"]
        self._completion_tokens += result["usage"]["completion_tokens"]
        return result["choices"][0]["message"]["content"]

    @property
    def usage(self):
        return {
            "total_tokens": self._prompt_tokens + self._completion_tokens,
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "cost": self.price_per_million_tokens * self._completion_tokens / 1e6
            if self.price_per_million_tokens
            else None,
        }
