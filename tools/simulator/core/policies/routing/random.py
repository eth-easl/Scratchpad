from ._base import BaseGlobalToLocalPolicy
from core.request import GenerationRequest
from core.engine import LLMEngine
from typing import Dict, List
import numpy as np

class RandomGTLPolicy(BaseGlobalToLocalPolicy):
    def __init__(self):
        """
        Random Global-to-Local routing policy.

        Assigns requests to engines randomly from the pool of engines that support
        the requested model. Simple but not optimal for load balancing.
        """
        super().__init__()
        self.engines = None

    def prepare(self, engines: Dict[str, List[LLMEngine]]) -> None:
        """
        Store reference to available engines.

        Args:
            engines: Dictionary mapping model names to lists of available engines
        """
        self.engines = engines

    def assign_requests(self, request: GenerationRequest) -> None:
        """
        Assign a request to a randomly selected compatible engine.

        Args:
            request: The GenerationRequest to be assigned
        """
        available_engines = self.engines[request.model]
        selected_engine = np.random.choice(available_engines, 1)[0]
        selected_engine.add_request(request)
