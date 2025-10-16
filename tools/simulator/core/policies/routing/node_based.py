from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import numpy as np
from ...node import ComputeNode
from ...request import GenerationRequest


class NodeBasedRoutingPolicy(ABC):
    """
    Abstract base class for node-based routing policies.

    These policies determine which node should handle each request
    when multiple nodes with different GPU configurations are available.
    """

    @abstractmethod
    def select_node(
        self, request: GenerationRequest, available_nodes: List[ComputeNode]
    ) -> Optional[ComputeNode]:
        """
        Select a node to handle the given request.

        Args:
            request: The request that needs to be assigned
            available_nodes: List of nodes that can serve the request's model

        Returns:
            Optional[ComputeNode]: Selected node, or None if no suitable node
        """
        pass

    @abstractmethod
    def prepare(self, nodes: Dict[str, List[ComputeNode]]) -> None:
        """
        Prepare the policy with available nodes.

        Args:
            nodes: Dictionary mapping model names to lists of nodes that can serve them
        """
        pass


class RandomNodePolicy(NodeBasedRoutingPolicy):
    """
    Random node selection policy.

    Randomly selects a node from the available nodes that can serve the request.
    Simple but provides load balancing across multiple nodes.
    """

    def __init__(self):
        """Initialize random node policy."""
        self.nodes_by_model = {}

    def prepare(self, nodes: Dict[str, List[ComputeNode]]) -> None:
        """
        Store reference to available nodes grouped by model.

        Args:
            nodes: Dictionary mapping model names to lists of nodes
        """
        self.nodes_by_model = nodes

    def select_node(
        self, request: GenerationRequest, available_nodes: List[ComputeNode]
    ) -> Optional[ComputeNode]:
        """
        Randomly select a node from available nodes.

        Args:
            request: The request that needs to be assigned
            available_nodes: List of nodes that can serve the request's model

        Returns:
            Optional[ComputeNode]: Randomly selected node, or None if no nodes available
        """
        if not available_nodes:
            return None

        return np.random.choice(available_nodes)


class LeastLoadedNodePolicy(NodeBasedRoutingPolicy):
    """
    Least loaded node selection policy.

    Selects the node with the lowest utilization (fewest running requests).
    Provides better load balancing than random selection.
    """

    def __init__(self):
        """Initialize least loaded node policy."""
        self.nodes_by_model = {}

    def prepare(self, nodes: Dict[str, List[ComputeNode]]) -> None:
        """
        Store reference to available nodes grouped by model.

        Args:
            nodes: Dictionary mapping model names to lists of nodes
        """
        self.nodes_by_model = nodes

    def select_node(
        self, request: GenerationRequest, available_nodes: List[ComputeNode]
    ) -> Optional[ComputeNode]:
        """
        Select the least loaded node from available nodes.

        Args:
            request: The request that needs to be assigned
            available_nodes: List of nodes that can serve the request's model

        Returns:
            Optional[ComputeNode]: Least loaded node, or None if no nodes available
        """
        if not available_nodes:
            return None

        # Find node with minimum running requests
        min_running_requests = min(
            node.running_requests.count() for node in available_nodes
        )
        least_loaded_nodes = [
            node
            for node in available_nodes
            if len(node.running_requests) == min_running_requests
        ]

        # If tie, randomly select from least loaded nodes
        return np.random.choice(least_loaded_nodes)


class GPUCountNodePolicy(NodeBasedRoutingPolicy):
    """
    Node selection policy based on GPU count.

    Prioritizes nodes with more available GPUs for the requested model,
    which can provide better throughput for large requests.
    """

    def __init__(self, prefer_more_gpus: bool = True):
        """
        Initialize GPU count based policy.

        Args:
            prefer_more_gpus: If True, prefer nodes with more GPUs; otherwise prefer fewer
        """
        self.nodes_by_model = {}
        self.prefer_more_gpus = prefer_more_gpus

    def prepare(self, nodes: Dict[str, List[ComputeNode]]) -> None:
        """
        Store reference to available nodes grouped by model.

        Args:
            nodes: Dictionary mapping model names to lists of nodes
        """
        self.nodes_by_model = nodes

    def select_node(
        self, request: GenerationRequest, available_nodes: List[ComputeNode]
    ) -> Optional[ComputeNode]:
        """
        Select node based on available GPU count for the model.

        Args:
            request: The request that needs to be assigned
            available_nodes: List of nodes that can serve the request's model

        Returns:
            Optional[ComputeNode]: Selected node, or None if no nodes available
        """
        if not available_nodes:
            return None

        # Count available GPUs for each node
        node_gpu_counts = []
        for node in available_nodes:
            available_gpus = node.get_available_gpus_for_model(request.model)
            node_gpu_counts.append((node, len(available_gpus)))

        # Sort by GPU count
        node_gpu_counts.sort(key=lambda x: x[1], reverse=self.prefer_more_gpus)

        # Select from nodes with maximum/minimum GPU count
        max_gpus = node_gpu_counts[0][1]
        best_nodes = [
            node for node, gpu_count in node_gpu_counts if gpu_count == max_gpus
        ]

        return np.random.choice(best_nodes)


class AffinityNodePolicy(NodeBasedRoutingPolicy):
    """
    Node selection policy based on request affinity.

    Tries to assign related requests to the same node to benefit from
    model caching and reduce model loading overhead.
    """

    def __init__(self):
        """Initialize affinity-based node policy."""
        self.nodes_by_model = {}
        self.request_affinity = {}  # Track which node handled similar requests

    def prepare(self, nodes: Dict[str, List[ComputeNode]]) -> None:
        """
        Store reference to available nodes grouped by model.

        Args:
            nodes: Dictionary mapping model names to lists of nodes
        """
        self.nodes_by_model = nodes

    def select_node(
        self, request: GenerationRequest, available_nodes: List[ComputeNode]
    ) -> Optional[ComputeNode]:
        """
        Select node based on request affinity.

        Args:
            request: The request that needs to be assigned
            available_nodes: List of nodes that can serve the request's model

        Returns:
            Optional[ComputeNode]: Selected node, or None if no nodes available
        """
        if not available_nodes:
            return None

        # Check if we have affinity information for this model
        model_affinity_key = request.model

        if model_affinity_key in self.request_affinity:
            preferred_node = self.request_affinity[model_affinity_key]
            if (
                preferred_node in available_nodes
                and preferred_node.get_available_gpus_for_model(request.model)
            ):
                return preferred_node

        # If no affinity or preferred node not available, fall back to least loaded
        min_running_requests = min(
            node.running_requests.count() for node in available_nodes
        )
        least_loaded_nodes = [
            node
            for node in available_nodes
            if len(node.running_requests) == min_running_requests
        ]

        selected_node = np.random.choice(least_loaded_nodes)

        # Update affinity
        self.request_affinity[model_affinity_key] = selected_node

        return selected_node
