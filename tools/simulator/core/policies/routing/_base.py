from abc import ABC, abstractmethod
from tools.simulator.request import GenerationRequest


class BaseGlobalToLocalPolicy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def prepare(self, engines):
        ...

    @abstractmethod
    def assign_requests(self, request: GenerationRequest):
        ...
