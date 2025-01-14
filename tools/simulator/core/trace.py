from dataclasses import dataclass
from typing import Optional


@dataclass
class TraceEvent:
    name: str
    cat: str
    ph: str
    pid: int
    tid: int

    ts: int  # in microseconds
    args: Optional[dict] = None
    dur: Optional[int] = None
