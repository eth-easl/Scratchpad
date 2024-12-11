import aiohttp
import asyncio
from typing import Dict


async def async_request(endpoint, req: Dict):
    async with aiohttp.ClientSession() as session:
        async with session.post(endpoint, json=req) as response:
            return await response.json()


async def make_requests(endpoint, reqs):
    return await asyncio.gather(*[async_request(endpoint, req) for req in reqs])
