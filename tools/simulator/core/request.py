import enum
from datetime import datetime
from typing import Dict


class REQ_STATUS(enum.Enum):
    PENDING = 1
    SCHEDULED = 2
    PREFILL = 3
    GENERATE = 4
    EXIT = 5


class GenerationRequest:
    def __init__(
        self,
        req_id: str,
        model: str,
        input_length: int,
        output_length: int,
        arrive_at: float,
    ):
        self.req_id = req_id
        self.model = model
        self.input_length = input_length
        self.output_length = output_length
        self.arrive_at = arrive_at
        self.status = REQ_STATUS.PENDING
        self.prefill_time = None

        self.generated_tokens = 0
        self.prefill_finished_at = None
        self.generation_finished_at = None

    def set_generation_finished_at(self, finished_at: float):
        self.generation_finished_at = finished_at

    def set_prefill_finished_at(self, finished_at: float):
        self.prefill_finished_at = finished_at

    def _prefill(self):
        self.status = REQ_STATUS.PREFILL

    def _decode(self) -> bool:
        self.generated_tokens += 1
        if self.generated_tokens == self.output_length:
            self._stop()
            return True
        return False

    def _stop(self):
        pass

    def __str__(self):
        return f"Request {self.req_id} for model {self.model} with input length {self.input_length} and output length {self.output_length} arrived at {self.arrive_at}"

    def __repr__(self) -> str:
        return self.__str__()

    def to_dict(self) -> Dict:
        return {
            "req_id": self.req_id,
            "model": self.model,
            "input_length": self.input_length,
            "output_length": self.output_length,
            "arrive_at": self.arrive_at,
            "prefill_time": self.prefill_time,
            "generated_tokens": self.generated_tokens,
            "prefill_finished_at": self.prefill_finished_at,
            "generation_finished_at": self.generation_finished_at,
        }
