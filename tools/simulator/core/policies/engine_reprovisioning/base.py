from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from ...engine import LLMEngine


class EngineReprovisioningPolicy(ABC):
    """
    Abstract base class for engine re-provisioning policies.

    This policy determines how to re-purpose existing engines when no suitable
    engine is available for a new model request.
    """

    @abstractmethod
    def select_engine_to_repurpose(
        self,
        target_model: str,
        all_engines: Dict[str, List[LLMEngine]],
        current_time: float,
    ) -> Optional[LLMEngine]:
        """
        Select an engine to re-purpose for the target model.

        Args:
            target_model: The model that needs an engine
            all_engines: Dictionary of all engines grouped by model
            current_time: Current simulation time

        Returns:
            Optional[LLMEngine]: Engine to re-purpose, or None if no engine can be re-purposed
        """
        pass

    @abstractmethod
    def get_reprovisioning_time(
        self, source_model: str, target_model: str, engine: LLMEngine
    ) -> float:
        """
        Calculate the time required to re-purpose an engine from source_model to target_model.

        Args:
            source_model: Current model on the engine
            target_model: Model to load on the engine
            engine: The engine being re-purposed

        Returns:
            float: Time in seconds required for re-provisioning
        """
        pass

    @abstractmethod
    def can_repurpose_engine(
        self, engine: LLMEngine, target_model: str, current_time: float
    ) -> bool:
        """
        Check if an engine can be re-purposed for the target model.

        Args:
            engine: The engine to check
            target_model: The target model
            current_time: Current simulation time

        Returns:
            bool: True if the engine can be re-purposed
        """
        pass
