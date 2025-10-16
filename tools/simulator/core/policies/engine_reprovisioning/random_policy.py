import random
import numpy as np
from typing import Optional, List, Dict, Any
from .base import EngineReprovisioningPolicy
from ...engine import LLMEngine
from ...env import calculate_model_loading_time


class RandomReprovisioningPolicy(EngineReprovisioningPolicy):
    """
    Random engine re-provisioning policy.

    This policy randomly selects an idle engine to re-purpose when no suitable
    engine is available for a new model request.
    """

    def __init__(
        self,
        model_sizes_gb: Dict[str, float] = None,
        default_disk_bandwidth_mbps: float = 1000,
        default_pcie_bandwidth_gbps: float = 32,
    ):
        """
        Initialize the random re-provisioning policy.

        Args:
            model_sizes_gb: Dictionary of model names to their sizes in GB
            default_disk_bandwidth_mbps: Default disk bandwidth for model loading
            default_pcie_bandwidth_gbps: Default PCIe bandwidth for model loading
        """
        self.model_sizes_gb = model_sizes_gb or {
            "meta-llama/Meta-Llama-3-70B-Instruct": 140.0,
        }
        self.default_disk_bandwidth_mbps = default_disk_bandwidth_mbps
        self.default_pcie_bandwidth_gbps = default_pcie_bandwidth_gbps

    def select_engine_to_repurpose(
        self,
        target_model: str,
        all_engines: Dict[str, List[LLMEngine]],
        current_time: float,
        engines_being_reprovisioned: set = None,
    ) -> Optional[LLMEngine]:
        """
        Randomly select an idle engine to re-purpose for the target model.

        Args:
            target_model: The model that needs an engine
            all_engines: Dictionary of all engines grouped by model
            current_time: Current simulation time
            engines_being_reprovisioned: Set of engine IDs currently being re-provisioned

        Returns:
            Optional[LLMEngine]: Randomly selected idle engine, or None if no idle engines
        """
        if engines_being_reprovisioned is None:
            engines_being_reprovisioned = set()

        idle_engines = []

        # Find all idle engines (not currently processing requests)
        for model_name, engines in all_engines.items():
            for engine in engines:
                # Skip engines that are already being re-provisioned
                if engine.engine_id in engines_being_reprovisioned:
                    continue
                if self.can_repurpose_engine(engine, target_model, current_time):
                    idle_engines.append((model_name, engine))

        if not idle_engines:
            return None

        # Randomly select an idle engine
        source_model, selected_engine = random.choice(idle_engines)
        return selected_engine

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
        # If the models are the same, no re-provisioning needed
        if source_model == target_model:
            return 0.0

        # Get model size for target model
        model_size_gb = self.model_sizes_gb.get(target_model, 140.0)

        # Calculate model loading time
        loading_time = calculate_model_loading_time(
            model_size_gb=model_size_gb,
            disk_bandwidth_mbps=self.default_disk_bandwidth_mbps,
            pcie_bandwidth_gbps=self.default_pcie_bandwidth_gbps,
            cache_available=False,
        )

        return loading_time

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
        # Engine can be re-purposed if it's idle (no waiting or running requests)
        return len(engine.waiting) == 0 and len(engine.running) == 0
