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
        self.global_engine = None  # Reference to global engine for re-provisioning

    def prepare(self, engines: Dict[str, List[LLMEngine]]) -> None:
        """
        Store reference to available engines.

        Args:
            engines: Dictionary mapping model names to lists of available engines
        """
        self.engines = engines

    def set_global_engine(self, global_engine) -> None:
        """
        Set reference to global engine for re-provisioning capabilities.

        Args:
            global_engine: Reference to the LLMGlobalEngine instance
        """
        self.global_engine = global_engine

    def assign_requests(self, request: GenerationRequest) -> None:
        """
        Assign a request to a randomly selected compatible engine.

        Args:
            request: The GenerationRequest to be assigned
        """
        # Check if there are available engines for the requested model
        if request.model not in self.engines or len(self.engines[request.model]) == 0:
            # No engines available for this model, trigger re-provisioning
            if self.global_engine and self.global_engine.can_create_engine_for_model(
                request.model
            ):
                print(
                    f"No engines available for {request.model}, attempting to create one..."
                )
                self.global_engine.create_engine_for_model(
                    request.model, self.global_engine.global_timer
                )
                # Request will be assigned in the next iteration after re-provisioning completes
            else:
                print(f"No engines available for {request.model} and cannot create one")
            return

        # Get available engines for the model
        available_engines = self.engines[request.model]

        # Filter out engines that are being re-provisioned
        idle_engines = [
            engine
            for engine in available_engines
            if engine.engine_id
            not in self.global_engine.engine_reprovisioning_in_progress
        ]

        if not idle_engines:
            # All engines are busy or being re-provisioned
            return

        selected_engine = np.random.choice(idle_engines, 1)[0]
        selected_engine.add_request(request)
