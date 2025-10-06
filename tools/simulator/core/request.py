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
        """
        Initialize a generation request.

        Args:
            req_id: Unique identifier for this request
            model: Name of the model that should process this request
            input_length: Number of input tokens (prompt length)
            output_length: Number of output tokens to generate
            arrive_at: Time when this request arrives in the simulation
        """
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
        """
        Set the time when generation completes for this request.

        Args:
            finished_at: Generation completion time in seconds
        """
        self.generation_finished_at = finished_at

    def set_prefill_finished_at(self, finished_at: float):
        """
        Set the time when prefill phase completes for this request.

        Args:
            finished_at: Prefill completion time in seconds
        """
        self.prefill_finished_at = finished_at

    def _prefill(self):
        """
        Mark request as entering prefill phase.
        Internal method used by LLMEngine.
        """
        self.status = REQ_STATUS.PREFILL

    def _decode(self) -> bool:
        """
        Process one decode step for this request.

        Returns:
            bool: True if generation is complete after this decode step
        """
        self.generated_tokens += 1
        if self.generated_tokens == self.output_length:
            self._stop()
            return True
        return False

    def _stop(self):
        """
        Mark request as stopped. Can be overridden in subclasses for cleanup.
        """
        pass

    def __str__(self):
        """
        String representation of the request for debugging.

        Returns:
            str: Human-readable description of the request
        """
        return f"Request {self.req_id} for model {self.model} with input length {self.input_length} and output length {self.output_length} arrived at {self.arrive_at}"

    def __repr__(self) -> str:
        """
        Official string representation of the request.

        Returns:
            str: String representation suitable for debugging
        """
        return self.__str__()

    def to_dict(self) -> Dict:
        """
        Convert request to dictionary for serialization.

        Returns:
            dict: Dictionary containing all request information for analysis
        """
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
