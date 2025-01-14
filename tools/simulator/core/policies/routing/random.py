from ._base import BaseGlobalToLocalPolicy
from adaml.simulator.request import GenerationRequest
from adaml.simulator.engine import LLMEngine
from typing import Dict, List
import numpy as np


class RandomGTLPolicy(BaseGlobalToLocalPolicy):
    def __init__(self):
        super().__init__()
        self.engines = None

    def prepare(self, engines: Dict[str, List[LLMEngine]]):
        self.engines = engines

    def assign_requests(self, request: GenerationRequest):
        available_engines = self.engines[request.model]
        # randomly pick one
        selected_engine = np.random.choice(available_engines, 1)[0]
        selected_engine.add_request(request)
