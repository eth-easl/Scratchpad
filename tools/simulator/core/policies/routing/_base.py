from abc import ABC, abstractmethod
from core.request import GenerationRequest


class BaseGlobalToLocalPolicy(ABC):
    def __init__(self):
        """
        Base class for request routing policies.

        Implementations should define how requests are assigned to specific engines
        based on various criteria like load balancing, model compatibility, etc.
        """
        pass

    @abstractmethod
    def prepare(self, engines):
        """
        Prepare the policy with the available engines.

        Args:
            engines: Dictionary mapping model names to lists of available engines
        """
        ...

    @abstractmethod
    def assign_requests(self, request: GenerationRequest):
        """
        Assign a request to a specific engine.

        Args:
            request: The GenerationRequest to be assigned
        """
        ...
