import json
from collections import deque, defaultdict
from typing import Dict, List, Optional, Set
from .node import ComputeNode, NodeConfig, GPUConfig
from .request import GenerationRequest
from .trace import TraceEvent
from .policies.routing import RandomNodePolicy
from .policies.node_reprovisioning import (
    NodeReprovisioningPolicy,
    RandomNodeReprovisioningPolicy,
)
from core.env import EnvironmentConfig, calculate_model_loading_time


class NodeGlobalEngine:
    """
    Global engine that manages multiple compute nodes.

    This is the new architecture that uses the Node abstraction instead of
    individual engines, providing more realistic infrastructure modeling
    with multiple GPUs per node.
    """

    def __init__(
        self,
        environment_config: Optional[EnvironmentConfig] = None,
        environment_changes: Optional[List] = None,
        routing_policy=None,
        node_reprovisioning_policy: Optional[NodeReprovisioningPolicy] = None,
        print_interval: float = 0.1,
    ):
        """
        Initialize the node-based global engine.

        Args:
            environment_config: Initial environment configuration with node definitions
            environment_changes: Dynamic environment changes (for future extension)
            routing_policy: Policy for routing requests to nodes
            node_reprovisioning_policy: Policy for re-provisioning nodes between models
            print_interval: Interval for printing progress updates in seconds
        """
        self.nodes = {}  # node_id -> ComputeNode
        self.pending_requests: deque[GenerationRequest] = deque()
        self.global_timer = 0.0
        self.supported_models: Set[str] = set()
        self._trace = []
        self.total_requests = 0

        # Environment configuration
        self.environment_config = environment_config
        self.environment_changes = environment_changes or []

        # Routing policy
        self.routing_policy = routing_policy or RandomNodePolicy()
        self.nodes_by_model = defaultdict(list)  # model_name -> list of nodes

        # Node re-provisioning policy
        self.node_reprovisioning_policy = (
            node_reprovisioning_policy or RandomNodeReprovisioningPolicy()
        )

        # Track model loading and re-provisioning
        self.node_reprovisioning_in_progress = {}  # node_id -> (end_time, target_model)

        # Progress tracking for timeout
        self.last_progress_time = 0.0
        self.simulation_timeout = 300.0  # 5 minutes timeout
        self.print_interval = print_interval
        self.last_print_time = 0.0

        # Initialize nodes from environment configuration
        if environment_config:
            self._initialize_nodes_from_config(environment_config)

    def _initialize_nodes_from_config(self, config: EnvironmentConfig):
        """
        Initialize nodes from environment configuration.

        Args:
            config: Environment configuration containing node settings
        """
        print(f"Initializing nodes from environment configuration...")

        total_loading_time = 0.0
        model_sizes_gb = {
            "meta-llama/Meta-Llama-3-70B-Instruct": 140.0,
            "meta-llama/Meta-Llama-3-8B-Instruct": 16.0,
            "meta-llama/Meta-Llama-3.1-8B-Instruct": 16.0,
        }

        # Prefer new node-based configuration over legacy GPU configuration
        if config.nodes:
            print(f"Using node-based configuration with {len(config.nodes)} nodes")
            for node_config_data in config.nodes:
                # Convert environment NodeConfig to node.NodeConfig
                gpus = []
                for i, (gpu_type, gpu_count, memory_gb) in enumerate(
                    zip(
                        node_config_data.gpu_types,
                        node_config_data.gpu_count_per_type,
                        node_config_data.memory_per_gpu_gb,
                    )
                ):
                    for gpu_idx in range(gpu_count):
                        gpu = GPUConfig(
                            gpu_id=gpu_idx, hardware_name=gpu_type, memory_gb=memory_gb
                        )
                        gpus.append(gpu)

                # Create NodeConfig for ComputeNode
                node_config = NodeConfig(
                    node_id=node_config_data.node_id,
                    name=node_config_data.name,
                    gpus=gpus,
                    total_memory_gb=sum(gpu.memory_gb for gpu in gpus)
                    * 1.2,  # 20% overhead
                    network_bandwidth_gbps=node_config_data.network_bandwidth_gbps,
                    disk_bandwidth_mbps=config.infrastructure.disk_bandwidth_mbps,
                    pcie_bandwidth_gbps=config.infrastructure.pcie_bandwidth_gbps,
                )

                node = ComputeNode(node_config)

                # For now, assume the first GPU type is the model to load
                # In a more sophisticated setup, we might load multiple models per node
                model_name = f"model-{node_config_data.gpu_types[0]}"  # Placeholder

                # Create GPU configurations for this node
                for i, (gpu_type, gpu_count, memory_gb) in enumerate(
                    zip(
                        node_config_data.gpu_types,
                        node_config_data.gpu_count_per_type,
                        node_config_data.memory_per_gpu_gb,
                    )
                ):
                    for gpu_idx in range(gpu_count):
                        gpu = GPUConfig(
                            gpu_id=gpu_idx, hardware_name=gpu_type, memory_gb=memory_gb
                        )

                        # Assign model name (simplified - would normally come from config)
                        model_to_load = (
                            "meta-llama/Meta-Llama-3-8B-Instruct"  # Default model
                        )
                        model_size_gb = model_sizes_gb.get(model_to_load, 16.0)

                        if config.model_loading.preload_models:
                            loading_time = 0.0
                            completion_time = 0.0
                        else:
                            loading_time = calculate_model_loading_time(
                                model_size_gb=model_size_gb,
                                disk_bandwidth_mbps=config.infrastructure.disk_bandwidth_mbps,
                                pcie_bandwidth_gbps=config.infrastructure.pcie_bandwidth_gbps,
                                cache_available=False,
                            )
                            completion_time = loading_time
                            total_loading_time = max(total_loading_time, loading_time)

                            # Add model loading trace event
                            loading_event = TraceEvent(
                                name=f"node-loading-{model_to_load}",
                                cat="node_model_loading",
                                ph="X",
                                pid=-node_config_data.node_id,
                                tid=gpu_idx,
                                ts=0,
                                dur=int(loading_time * 1_000_000),
                                args={
                                    "model": model_to_load,
                                    "node_id": node_config_data.node_id,
                                    "gpu_id": gpu_idx,
                                    "hardware": gpu_type,
                                    "model_size_gb": model_size_gb,
                                    "loading_time_s": loading_time,
                                },
                            )
                            self._trace.append(loading_event)

                        # Add GPU to node
                        node.add_gpu(gpu, model_to_load, completion_time)

                self.nodes[node_config_data.node_id] = node
                self.supported_models.add(model_to_load)
                self.nodes_by_model[model_to_load].append(node)

        elif config.gpus:
            print(f"Using legacy GPU configuration with {len(config.gpus)} GPU types")
            # Legacy: Create one node per GPU for backward compatibility
            node_id = 1
            for gpu_config in config.gpus:
                # Create a node with a single GPU
                gpu = GPUConfig(
                    gpu_id=0,  # Single GPU per node
                    hardware_name=gpu_config.name,
                    memory_gb=40.0
                    if "A100" in gpu_config.name
                    else 80.0,  # Estimate GPU memory
                )

                node_config = NodeConfig(
                    node_id=node_id,
                    name=f"node-{node_id}",
                    gpus=[gpu],
                    total_memory_gb=gpu.memory_gb * 1.2,  # 20% overhead
                    network_bandwidth_gbps=config.infrastructure.network_bandwidth_gbps,
                    disk_bandwidth_mbps=config.infrastructure.disk_bandwidth_mbps,
                    pcie_bandwidth_gbps=config.infrastructure.pcie_bandwidth_gbps,
                )

                node = ComputeNode(node_config)

                # Add GPU and load model
                model_size_gb = model_sizes_gb.get(gpu_config.model, 1.0)

                if config.model_loading.preload_models:
                    # Preload the model
                    loading_time = 0.0
                    completion_time = 0.0
                else:
                    # Calculate loading time
                    loading_time = calculate_model_loading_time(
                        model_size_gb=model_size_gb,
                        disk_bandwidth_mbps=config.infrastructure.disk_bandwidth_mbps,
                        pcie_bandwidth_gbps=config.infrastructure.pcie_bandwidth_gbps,
                        cache_available=False,
                    )
                    completion_time = loading_time
                    total_loading_time = max(total_loading_time, loading_time)
                    print(
                        f"  Model {gpu_config.model} loading time: {loading_time:.2f}s"
                    )

                    # Add model loading trace event
                    loading_event = TraceEvent(
                        name=f"node-loading-{gpu_config.model}",
                        cat="node_model_loading",
                        ph="X",
                        pid=-node_id,  # Use negative node_id for system-level events
                        tid=0,
                        ts=0,
                        dur=int(loading_time * 1_000_000),
                        args={
                            "model": gpu_config.model,
                            "node_id": node_id,
                            "hardware": gpu_config.name,
                            "model_size_gb": model_size_gb,
                            "loading_time_s": loading_time,
                        },
                    )
                    self._trace.append(loading_event)

                # Add GPU to node
                node.add_gpu(gpu, gpu_config.model, completion_time)

                self.nodes[node_id] = node
                self.supported_models.add(gpu_config.model)
                self.nodes_by_model[gpu_config.model].append(node)

                node_id += 1

        else:
            print("Warning: No nodes or GPUs found in environment configuration")
            return

        # Prepare routing policy
        self.routing_policy.prepare(self.nodes_by_model)

        if total_loading_time > 0:
            print(f"Total initialization time: {total_loading_time:.2f}s")
            self.global_timer = total_loading_time
            print(f"Simulation starts at time: {self.global_timer:.2f}s")

        print(f"Initialized {len(self.nodes)} nodes")

    def add_node(
        self, node_config: NodeConfig, model_name: str = None, preload: bool = True
    ) -> int:
        """
        Add a new node to the global engine.

        Args:
            node_config: Configuration for the new node
            model_name: Model to load on the node
            preload: Whether to preload the model

        Returns:
            int: ID of the newly created node
        """
        node_id = max(self.nodes.keys()) + 1 if self.nodes else 1
        node_config.node_id = node_id
        node_config.name = f"node-{node_id}"

        node = ComputeNode(node_config)

        if model_name:
            # Add GPUs and load model
            for gpu_config in node_config.gpus:
                loading_time = 0.0 if preload else None
                node.add_gpu(gpu_config, model_name, loading_time)

            self.supported_models.add(model_name)
            self.nodes_by_model[model_name].append(node)

        self.nodes[node_id] = node
        self.routing_policy.prepare(self.nodes_by_model)

        return node_id

    def load_requests(self, requests: List[GenerationRequest]):
        """
        Load a batch of requests into the pending queue.

        Args:
            requests: List of GenerationRequest objects to be processed
        """
        for req in requests:
            self.pending_requests.append(req)
            self.total_requests += 1

    def get_nodes_for_model(self, model_name: str) -> List[ComputeNode]:
        """
        Get all nodes that can serve the given model.

        Args:
            model_name: Model name

        Returns:
            List[ComputeNode]: Nodes that can serve the model
        """
        return self.nodes_by_model.get(model_name, [])

    def start(self):
        """
        Start the simulation loop with node-based processing.

        The simulation proceeds by:
        1. Processing all nodes that are ready at the current time
        2. Advancing global time to the next earliest event
        3. Assigning newly arrived requests to nodes
        4. Continuing until all requests are processed
        """
        print(f"Total requests: {self.total_requests}")
        print(f"Available nodes: {len(self.nodes)}")
        time_queue = set()

        # Track progress
        previous_finished = 0

        while True:
            made_progress = False

            # Process each node
            for node_id, node in self.nodes.items():
                node_events, completed_requests, next_time = node.step(
                    self.global_timer
                )

                if node_events:
                    self._trace.extend(node_events)
                    if completed_requests:
                        made_progress = True

                time_queue.add(next_time)

            # Check for new request assignments
            self.assign_requests_to_nodes(self.global_timer)

            # Add re-provisioning completion times to time queue
            for end_time, target_model in self.node_reprovisioning_in_progress.values():
                time_queue.add(end_time)

            # If no meaningful time advances (all times are <= current_time),
            # add a small increment to avoid infinite loops
            valid_times = [t for t in time_queue if t > self.global_timer]
            if valid_times:
                self.global_timer = min(valid_times)
                time_queue.discard(self.global_timer)
            else:
                # If only re-provisioning times exist and they're all in the future,
                # jump to the earliest re-provisioning completion
                if self.node_reprovisioning_in_progress:
                    earliest_reprovisioning = min(
                        end_time
                        for end_time, _ in self.node_reprovisioning_in_progress.values()
                    )
                    if earliest_reprovisioning > self.global_timer:
                        self.global_timer = earliest_reprovisioning
                    else:
                        self.global_timer += 0.001
                else:
                    self.global_timer += 0.001  # Small advance

            # Update progress tracking
            current_finished = sum(
                len(node.completed_requests) for node in self.nodes.values()
            )
            if current_finished > previous_finished:
                made_progress = True
                self.last_progress_time = self.global_timer
                previous_finished = current_finished

            # Check for timeout
            if made_progress:
                self.last_progress_time = self.global_timer
            elif self.global_timer - self.last_progress_time > self.simulation_timeout:
                print(
                    f"\nSimulation timeout: No progress for {self.simulation_timeout}s"
                )
                print(f"Finished: {self.finished_percentage:.2f}% of requests")
                break

            # Print progress with interval control
            if self.global_timer - self.last_print_time >= self.print_interval:
                active_nodes = sum(
                    1 for node in self.nodes.values() if not node.is_idle
                )
                print(
                    f"Finished: {self.finished_percentage:.2f}%, "
                    f"Time: {self.global_timer:.2f}, "
                    f"Active Nodes: {active_nodes}/{len(self.nodes)}",
                    end="\r",
                )
                self.last_print_time = self.global_timer

            if not self.has_remaining_requests():
                break

        print(f"\nSimulation completed at time: {self.global_timer:.2f}")

    def assign_requests_to_nodes(self, current_time: float):
        """
        Assign pending requests to available nodes.

        Args:
            current_time: Current simulation time
        """
        if not self.pending_requests:
            return

        # Check if any node re-provisioning has completed
        completed_reprovisioning = []
        for node_id, (
            end_time,
            target_model,
        ) in self.node_reprovisioning_in_progress.items():
            if current_time >= end_time:
                completed_reprovisioning.append(node_id)
                # Update node to serve the new model
                if node_id in self.nodes:
                    node = self.nodes[node_id]
                    # Clear old model mappings and add new ones
                    for model, nodes_list in self.nodes_by_model.items():
                        if node in nodes_list:
                            nodes_list.remove(node)

                    # Update the node's loaded models
                    node.loaded_models.clear()
                    node.loaded_models[target_model] = set(
                        gpu.gpu_id for gpu in node.config.gpus
                    )

                    # Update routing tables
                    self.nodes_by_model[target_model].append(node)
                    self.supported_models.add(target_model)
                    print(
                        f"T: {current_time:.2f} Node {node_id} re-provisioned to serve {target_model}"
                    )

        # Remove completed re-provisioning from tracking
        for node_id in completed_reprovisioning:
            del self.node_reprovisioning_in_progress[node_id]

        # Get requests that have arrived by current_time
        ready_requests = [
            req for req in self.pending_requests if req.arrive_at <= current_time
        ]

        for request in ready_requests:
            # Find nodes that can serve this model
            available_nodes = self.get_nodes_for_model(request.model)

            # Filter nodes that have capacity and are not being re-provisioned
            nodes_with_capacity = []
            for node in available_nodes:
                if (
                    node.can_serve_model(request.model)
                    and node.node_id not in self.node_reprovisioning_in_progress
                ):
                    nodes_with_capacity.append(node)

            if nodes_with_capacity:
                # Use routing policy to select node
                selected_node = self.routing_policy.select_node(
                    request, nodes_with_capacity
                )

                if selected_node and selected_node.add_request(request):
                    self.pending_requests.remove(request)
                    print(
                        f"T: {current_time:.2f} Assigned request {request.req_id} "
                        f"to node {selected_node.node_id}"
                    )
                else:
                    # No suitable node found, request stays in queue
                    pass
            else:
                # No nodes can serve this model, try to re-provision a node
                nodes_being_reprovisioned = set(
                    self.node_reprovisioning_in_progress.keys()
                )
                selected_node = (
                    self.node_reprovisioning_policy.select_node_to_repurpose(
                        request.model,
                        self.nodes,
                        current_time,
                        nodes_being_reprovisioned,
                    )
                )

                if selected_node:
                    # Calculate re-provisioning time
                    current_models = (
                        list(selected_node.loaded_models.keys())
                        if selected_node.loaded_models
                        else []
                    )
                    reprovisioning_time = (
                        self.node_reprovisioning_policy.get_reprovisioning_time(
                            current_models, request.model, selected_node
                        )
                    )

                    # Start re-provisioning
                    end_time = current_time + reprovisioning_time
                    self.node_reprovisioning_in_progress[selected_node.node_id] = (
                        end_time,
                        request.model,
                    )

                    # Add re-provisioning trace event
                    reprovision_event = TraceEvent(
                        name=f"node-reprovisioning-{selected_node.node_id}",
                        cat="node_reprovisioning",
                        ph="X",
                        pid=-selected_node.node_id,
                        tid=0,
                        ts=int(current_time * 1_000_000),
                        dur=int(reprovisioning_time * 1_000_000),
                        args={
                            "node_id": selected_node.node_id,
                            "source_models": current_models,
                            "target_model": request.model,
                            "reprovisioning_time_s": reprovisioning_time,
                        },
                    )
                    self._trace.append(reprovision_event)

                    print(
                        f"T: {current_time:.2f} Starting re-provisioning node {selected_node.node_id} "
                        f"from {current_models} to {request.model} (time: {reprovisioning_time:.2f}s)"
                    )
                else:
                    # No nodes available for re-provisioning
                    print(
                        f"T: {current_time:.2f} No available nodes for model {request.model} "
                        f"(no nodes available for re-provisioning)"
                    )

    def has_remaining_requests(self) -> bool:
        """
        Check if there are any requests remaining to be processed.

        Returns:
            bool: True if there are pending or running requests
        """
        if self.pending_requests:
            return True

        # Check if any node re-provisioning is in progress
        if self.node_reprovisioning_in_progress:
            return True

        for node in self.nodes.values():
            if node.waiting_requests or node.running_requests:
                return True

        return False

    @property
    def finished_percentage(self) -> float:
        """
        Calculate the percentage of requests that have been completed.

        Returns:
            float: Percentage (0-100) of finished requests
        """
        total_finished = sum(
            len(node.completed_requests) for node in self.nodes.values()
        )
        if self.total_requests == 0:
            return 100.0
        return 100 * total_finished / self.total_requests

    @property
    def trace(self) -> List:
        """
        Get the Chrome trace format events for all processed requests.

        Returns:
            List[TraceEvent]: Events that can be loaded into Chrome tracing
        """
        return self._trace

    @property
    def requests_stats(self) -> List[Dict]:
        """
        Get detailed statistics for all completed requests.

        Returns:
            List[dict]: Statistics including timing information for each completed request
        """
        stats = []
        for node in self.nodes.values():
            for request in node.completed_requests:
                stats.append(request.to_dict())
        return stats

    @property
    def summary(self) -> List[Dict]:
        """
        Calculate performance summary statistics for all completed requests.

        Returns:
            List[dict]: Performance metrics including latency, throughput, and percentiles
        """
        stats = self.requests_stats
        if not stats:
            return []

        avg_latency = sum(
            [x["generation_finished_at"] - x["arrive_at"] for x in stats]
        ) / len(stats)
        throughput = len(stats) / max([x["generation_finished_at"] for x in stats])
        p90_latency = sorted(
            [x["generation_finished_at"] - x["arrive_at"] for x in stats]
        )[int(len(stats) * 0.9)]
        p95_latency = sorted(
            [x["generation_finished_at"] - x["arrive_at"] for x in stats]
        )[int(len(stats) * 0.95)]
        avg_time_to_first_token = sum(
            [x["prefill_finished_at"] - x["arrive_at"] for x in stats]
        ) / len(stats)
        p90_ttft = sorted([x["prefill_finished_at"] - x["arrive_at"] for x in stats])[
            int(len(stats) * 0.9)
        ]
        p95_ttft = sorted([x["prefill_finished_at"] - x["arrive_at"] for x in stats])[
            int(len(stats) * 0.95)
        ]

        summaries = [
            {"Metric": "Avg E2E-Latency (s)", "Value": avg_latency},
            {"Metric": "Avg TTFT (s)", "Value": avg_time_to_first_token},
            {"Metric": "Throughput (req/s)", "Value": throughput},
            {"Metric": "P90 Latency (s)", "Value": p90_latency},
            {"Metric": "P95 Latency (s)", "Value": p95_latency},
            {"Metric": "P90 TTFT (s)", "Value": p90_ttft},
            {"Metric": "P95 TTFT (s)", "Value": p95_ttft},
        ]
        return summaries

    def get_node_status(self) -> Dict[int, Dict]:
        """
        Get status information for all nodes.

        Returns:
            Dict[int, Dict]: Node ID -> status information
        """
        return {node_id: node.get_status() for node_id, node in self.nodes.items()}
