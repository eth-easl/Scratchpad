import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from scratchpad.scheduler.stats import Stats


class SupportsMetricsInfo(Protocol):
    def metrics_info(self) -> Dict[str, str]:
        ...


class StatLoggerBase(ABC):
    """Base class for StatLogger."""

    def __init__(self, local_interval: float) -> None:
        # Tracked stats over current local logging interval.
        self.num_prompt_tokens: List[int] = []
        self.num_generation_tokens: List[int] = []
        self.last_local_log = time.time()
        self.local_interval = local_interval

    @abstractmethod
    def log(self, stats: "Stats") -> None:
        raise NotImplementedError

    @abstractmethod
    def info(self, type: str, obj: SupportsMetricsInfo) -> None:
        raise NotImplementedError

    def maybe_update_spec_decode_metrics(self, stats: "Stats"):
        """Save spec decode metrics (since they are unlikely
        to be emitted at same time as log interval)."""
        if stats.spec_decode_metrics is not None:
            self.spec_decode_metrics = stats.spec_decode_metrics
