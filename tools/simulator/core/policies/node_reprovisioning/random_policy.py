import random
from typing import Dict, List, Optional
from .base import NodeReprovisioningPolicy
from ...node import ComputeNode
from core.env import calculate_model_loading_time


class RandomNodeReprovisioningPolicy(NodeReprovisioningPolicy):
    """
    Random node re-provisioning policy.

    This policy randomly selects an idle node to re-purpose when no suitable
    node is available for a new model request.
    """

    def __init__(
        self,
        model_sizes_gb: Dict[str, float] = None,
        default_disk_bandwidth_mbps: float = 1000,
        default_pcie_bandwidth_gbps: float = 32,
    ):
        """
        Initialize the random node re-provisioning policy.

        Args:
            model_sizes_gb: Dictionary of model names to their sizes in GB
            default_disk_bandwidth_mbps: Default disk bandwidth for model loading
            default_pcie_bandwidth_gbps: Default PCIe bandwidth for model loading
        """
        self.model_sizes_gb = model_sizes_gb or {
            "meta-llama/Meta-Llama-3-70B-Instruct": 140.0,
            "meta-llama/Meta-Llama-3-8B-Instruct": 16.0,
            "meta-llama/Meta-Llama-3.1-8B-Instruct": 16.0,
        }
        self.default_disk_bandwidth_mbps = default_disk_bandwidth_mbps
        self.default_pcie_bandwidth_gbps = default_pcie_bandwidth_gbps

    def select_node_to_repurpose(
        self,
        target_model: str,
        all_nodes: Dict[int, ComputeNode],
        current_time: float,
        nodes_being_reprovisioned: set = None,
    ) -> Optional[ComputeNode]:
        """
        Randomly select an idle node to re-purpose for the target model.

        Args:
            target_model: The model that needs a node
            all_nodes: Dictionary of all available nodes
            current_time: Current simulation time
            nodes_being_reprovisioned: Set of node IDs currently being re-provisioned

        Returns:
            Optional[ComputeNode]: Randomly selected idle node, or None if no idle nodes
        """
        if nodes_being_reprovisioned is None:
            nodes_being_reprovisioned = set()

        idle_nodes = []

        print(
            f"T: {current_time:.2f} Looking for nodes to re-purpose for {target_model}"
        )
        print(f"T: {current_time:.2f} Total nodes available: {len(all_nodes)}")
        print(
            f"T: {current_time:.2f} Nodes being re-provisioned: {nodes_being_reprovisioned}"
        )

        # Find all idle nodes (not currently processing requests)
        for node_id, node in all_nodes.items():
            # Skip nodes that are already being re-provisioned
            if node_id in nodes_being_reprovisioned:
                continue

            can_repurpose = self.can_repurpose_node(node, target_model, current_time)
            node_status = f"running_requests={len(node.running_requests)}, loading={len(node.model_loading_in_progress)}"

            print(
                f"T: {current_time:.2f} Node {node_id}: can_repurpose={can_repurpose}, status={node_status}"
            )

            if can_repurpose:
                idle_nodes.append((node_id, node))

        print(
            f"T: {current_time:.2f} Found {len(idle_nodes)} idle nodes for re-provisioning"
        )

        if not idle_nodes:
            return None

        # Randomly select an idle node
        node_id, selected_node = random.choice(idle_nodes)
        print(f"T: {current_time:.2f} Selected node {node_id} for re-provisioning")
        return selected_node

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
        # If the models are the same, no re-provisioning needed
        if target_model in source_models:
            return 0.0

        # Get model size for target model
        model_size_gb = self.model_sizes_gb.get(target_model, 140.0)

        # Calculate model loading time
        loading_time = calculate_model_loading_time(
            model_size_gb=model_size_gb,
            disk_bandwidth_mbps=getattr(node.config, "disk_bandwidth_mbps", 1000),
            pcie_bandwidth_gbps=getattr(node.config, "pcie_bandwidth_gbps", 32),
            cache_available=False,
        )

        return loading_time

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
        # Node can be re-purposed if it's idle (no running requests)
        # and not currently loading models
        return (
            len(node.running_requests) == 0 and len(node.model_loading_in_progress) == 0
        )
