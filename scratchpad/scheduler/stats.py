from typing import List
from dataclasses import dataclass
from scratchpad.utils.platforms.cuda import get_gpu_utilization


@dataclass
class Stats:
    """Created by LLMEngine for use by StatLogger."""

    now: float
    generation_throughput: float
    running_requests: int
    queued_requests: int
    token_usage: float
    used_token_pool: int
    gpu_utilization: float = get_gpu_utilization(0)
