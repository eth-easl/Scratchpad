from typing import List
from dataclasses import dataclass


@dataclass
class Stats:
    """Created by LLMEngine for use by StatLogger."""

    now: float
    generation_throughput: float
    running_requests: int
    queued_requests: int
    token_usage: float
    used_token_pool: int
