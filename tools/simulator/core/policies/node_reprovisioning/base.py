from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from ...node import ComputeNode


class NodeReprovisioningPolicy(ABC):
    """
    Abstract base class for node re-provisioning policies.

    These policies determine how to re-purpose nodes when no suitable
    node is available for a new model request.
    """

    @abstractmethod
    def select_node_to_repurpose(
        self, target_model: str, all_nodes: Dict[int, ComputeNode], current_time: float
    ) -> Optional[ComputeNode]:
        """
        Select a node to re-purpose for the target model.

        Args:
            target_model: The model that needs a node
            all_nodes: Dictionary of all available nodes
            current_time: Current simulation time

        Returns:
            Optional[ComputeNode]: Node to re-purpose, or None if no node can be re-purposed
        """
        pass

    @abstractmethod
    def get_reprovisioning_time(
        self, source_models: List[str], target_model: str, node: ComputeNode
    ) -> float:
        """
        Calculate the time required to re-purpose a node from source models to target model.

        Args:
            source_models: Current models on the node
            target_model: Model to load on the node
            node: The node being re-purposed

        Returns:
            float: Time in seconds required for re-provisioning
        """
        pass

    @abstractmethod
    def can_repurpose_node(
        self, node: ComputeNode, target_model: str, current_time: float
    ) -> bool:
        """
        Check if a node can be re-purposed for the target model.

        Args:
            node: The node to check
            target_model: The target model
            current_time: Current simulation time

        Returns:
            bool: True if the node can be re-purposed
        """
        pass
