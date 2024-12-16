import os
import anyio
import requests
import aiohttp
import asyncio
from typing import Dict, Optional


async def async_request(endpoint, req: Dict):
    async with aiohttp.ClientSession() as session:
        async with session.post(endpoint, json=req) as response:
            return await response.json()


async def make_requests(endpoint, reqs):
    return await asyncio.gather(*[async_request(endpoint, req) for req in reqs])


class LLM:
    def __init__(
        self,
        model: str,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        if not endpoint:
            endpoint = os.environ.get("RC_API_BASE", None)
        if not api_key:
            api_key = os.environ.get("RC_API_KEY", None)
        if not endpoint or not api_key:
            raise ValueError("API key or endpoint not found")
        if not system_prompt:
            system_prompt = "You are a helpful assistant."
        self.model = model
        self.endpoint = endpoint + "/chat/completions"
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

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
                self.endpoint,
                headers=self.headers,
                json=data,
            ).json()
        except Exception as e:
            print(e)
            return None
        return res["choices"][0]["message"]["content"]
