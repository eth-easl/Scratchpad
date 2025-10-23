from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class TraceEvent:
    name: str
    cat: str
    ph: str
    pid: Union[int, str]
    tid: Union[int, str]

    ts: int  # in microseconds
    args: Optional[dict] = None
    dur: Optional[int] = None
