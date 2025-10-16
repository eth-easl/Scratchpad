from collections import deque, defaultdict
from typing import Dict, List, Deque, Optional, Set
from dataclasses import dataclass
from .engine import LLMEngine
from .request import GenerationRequest
from .trace import TraceEvent
from core.env import calculate_model_loading_time


@dataclass
class GPUConfig:
    """Configuration for a single GPU in a node."""

    gpu_id: int
    hardware_name: str
    memory_gb: float
    compute_capability: str = "8.0"  # Default for modern GPUs


@dataclass
class NodeConfig:
    """Configuration for a compute node."""

    node_id: int
    name: str
    gpus: List[GPUConfig]
    total_memory_gb: float
    network_bandwidth_gbps: float
    disk_bandwidth_mbps: float
    pcie_bandwidth_gbps: float


class ComputeNode:
    """
    A compute node containing multiple GPUs that can serve models.

    The Node abstraction represents a physical server with multiple GPUs,
    managing resource allocation, model loading, and request scheduling
    across the GPUs in the node.
    """

    def __init__(self, config: NodeConfig):
        """
        Initialize a compute node with the given configuration.

        Args:
            config: Node configuration containing GPU specifications and resources
        """
        self.config = config
        self.node_id = config.node_id
        self.name = config.name

        # GPU management
        self.gpus = {}  # gpu_id -> LLMEngine
        self.gpu_memory_usage = {}  # gpu_id -> used_memory_gb
        self.available_gpus = set(gpu_config.gpu_id for gpu_config in config.gpus)

        # Model management
        self.loaded_models = {}  # model_name -> Set[gpu_ids]
        self.model_loading_in_progress = {}  # model_name -> {gpu_ids, end_time}

        # Request management
        self.waiting_requests: Deque[GenerationRequest] = deque()
        self.running_requests: List[GenerationRequest] = []
        self.completed_requests: List[GenerationRequest] = []

        # Scheduling
        self.current_time = 0.0
        self.next_available_time = 0.0

        # Memory planning (node-level)
        self.node_memory_planner = NodeMemoryPlanner(config)

    def add_gpu(
        self,
        gpu_config: GPUConfig,
        model_name: str = None,
        w_bit: int = 4,
        a_bit: int = 4,
        kv_bit: int = 4,
    ) -> bool:
        """
        Add a GPU to the node.

        Args:
            gpu_config: GPU configuration
            model_name: Model to load on this GPU (optional)
            w_bit: Weight precision
            a_bit: Activation precision
            kv_bit: KV cache precision

        Returns:
            bool: True if GPU was added successfully
        """
        if gpu_config.gpu_id not in self.available_gpus:
            return False

        # Create engine for this GPU
        from internal.configs.hardware_params import hardware_params

        engine = LLMEngine(
            f"{self.node_id}-{gpu_config.gpu_id}",
            model_name or "placeholder",
            gpu_config.hardware_name,
            w_bit,
            a_bit,
            kv_bit,
        )

        self.gpus[gpu_config.gpu_id] = engine
        self.available_gpus.discard(gpu_config.gpu_id)
        self.gpu_memory_usage[gpu_config.gpu_id] = 0.0

        if model_name:
            self.load_model_on_gpu(model_name, gpu_config.gpu_id)

        return True

    def load_model_on_gpu(
        self, model_name: str, gpu_id: int, preload_time: float = 0.0
    ) -> float:
        """
        Load a model on a specific GPU.

        Args:
            model_name: Model to load
            gpu_id: GPU ID to load the model on
            preload_time: If 0, calculate loading time; otherwise use this time

        Returns:
            float: Time when model loading completes
        """
        if gpu_id not in self.gpus:
            raise ValueError(f"GPU {gpu_id} not found in node {self.node_id}")

        # Calculate loading time if not preloaded
        if preload_time == 0:
            model_sizes_gb = {
                "meta-llama/Meta-Llama-3-70B-Instruct": 140.0,
                "meta-llama/Meta-Llama-3-8B-Instruct": 16.0,
                "meta-llama/Meta-Llama-3.1-8B-Instruct": 16.0,
            }

            model_size_gb = model_sizes_gb.get(model_name, 140.0)
            loading_time = calculate_model_loading_time(
                model_size_gb=model_size_gb,
                disk_bandwidth_mbps=self.config.disk_bandwidth_mbps,
                pcie_bandwidth_gbps=self.config.pcie_bandwidth_gbps,
                cache_available=False,
            )
        else:
            loading_time = preload_time

        completion_time = self.current_time + loading_time

        # Track model loading
        if model_name not in self.model_loading_in_progress:
            self.model_loading_in_progress[model_name] = {
                "gpu_ids": set(),
                "end_time": completion_time,
            }

        self.model_loading_in_progress[model_name]["gpu_ids"].add(gpu_id)

        print(
            f"T: {self.current_time:.2f} Loading {model_name} on GPU {gpu_id} "
            f"(node {self.node_id}, completes at {completion_time:.2f})"
        )

        return completion_time

    def check_model_loading_completions(self):
        """Check for completed model loading and update GPU availability."""
        completed_models = []

        for model_name, loading_info in list(self.model_loading_in_progress.items()):
            if self.current_time >= loading_info["end_time"]:
                completed_models.append(model_name)

                # Update loaded models tracking
                if model_name not in self.loaded_models:
                    self.loaded_models[model_name] = set()
                self.loaded_models[model_name].update(loading_info["gpu_ids"])

                print(
                    f"T: {self.current_time:.2f} Model {model_name} loaded on GPUs "
                    f"{loading_info['gpu_ids']} in node {self.node_id}"
                )

        # Remove completed loading from tracking
        for model_name in completed_models:
            del self.model_loading_in_progress[model_name]

    def can_serve_model(self, model_name: str) -> bool:
        """
        Check if the node can serve the given model.

        Args:
            model_name: Model to check

        Returns:
            bool: True if model is loaded on at least one GPU
        """
        return (
            model_name in self.loaded_models and len(self.loaded_models[model_name]) > 0
        ) or (model_name in self.model_loading_in_progress)

    def get_available_gpus_for_model(self, model_name: str) -> List[int]:
        """
        Get list of GPU IDs that can serve the given model.

        Args:
            model_name: Model to check

        Returns:
            List[int]: Available GPU IDs for the model
        """
        if model_name not in self.loaded_models:
            return []

        # Find GPUs that are not currently busy (no running requests)
        available_gpus = []
        for gpu_id in self.loaded_models[model_name]:
            if gpu_id in self.gpus:
                engine = self.gpus[gpu_id]
                if len(engine.running) == 0:  # GPU is not busy
                    available_gpus.append(gpu_id)

        return available_gpus

    def add_request(self, request: GenerationRequest) -> bool:
        """
        Add a request to the node.

        Args:
            request: Request to add

        Returns:
            bool: True if request was accepted, False if node cannot serve the model
        """
        if not self.can_serve_model(request.model):
            return False

        self.waiting_requests.append(request)
        return True

    def step(self, current_time: float):
        """
        Process one time step for this node.

        Args:
            current_time: Current simulation time

        Returns:
            tuple: (events, completed_requests, next_time)
        """
        self.current_time = current_time
        events = []
        completed_requests = []

        # Check for model loading completions
        self.check_model_loading_completions()

        # Try to assign waiting requests to available GPUs
        self._assign_waiting_requests()

        # Process requests on each GPU
        next_times = []
        for gpu_id, engine in self.gpus.items():
            engine_event, finished_reqs, next_time, memory_event = engine.step(
                current_time
            )

            if engine_event is not None:
                events.extend(engine_event)
                events.append(memory_event)

            if finished_reqs:
                completed_requests.extend(finished_reqs)
                # Move finished requests from running to completed
                for finished_req in finished_reqs:
                    if finished_req in self.running_requests:
                        self.running_requests.remove(finished_req)
                        self.completed_requests.append(finished_req)
                        print(
                            f"T: {current_time:.2f} Request {finished_req.req_id} completed on GPU {gpu_id} (node {self.node_id})"
                        )

            next_times.append(next_time)

        # Determine next time step
        if next_times:
            next_time = min(next_times)
        else:
            next_time = current_time + 0.001  # Small advance

        return events, completed_requests, next_time

    def _assign_waiting_requests(self):
        """Assign waiting requests to available GPUs."""
        if not self.waiting_requests:
            return

        # Create a copy of waiting requests to avoid modification during iteration
        requests_to_assign = list(self.waiting_requests)

        for request in requests_to_assign:
            available_gpus = self.get_available_gpus_for_model(request.model)

            if available_gpus:
                # Assign to the first available GPU (could implement smarter selection)
                gpu_id = available_gpus[0]
                engine = self.gpus[gpu_id]

                engine.add_request(request)
                self.waiting_requests.remove(request)
                self.running_requests.append(request)

                print(
                    f"T: {self.current_time:.2f} Assigned request {request.req_id} "
                    f"to GPU {gpu_id} (node {self.node_id})"
                )

    @property
    def is_idle(self) -> bool:
        """Check if the node is idle (no running requests and no loading in progress)."""
        return (
            len(self.running_requests) == 0 and len(self.model_loading_in_progress) == 0
        )

    @property
    def utilization(self) -> float:
        """Calculate node utilization (0.0 to 1.0)."""
        if not self.gpus:
            return 0.0

        busy_gpus = sum(1 for engine in self.gpus.values() if len(engine.running) > 0)
        return busy_gpus / len(self.gpus)

    def get_status(self) -> Dict:
        """Get current node status for monitoring."""
        return {
            "node_id": self.node_id,
            "name": self.name,
            "total_gpus": len(self.gpus),
            "busy_gpus": sum(
                1 for engine in self.gpus.values() if len(engine.running) > 0
            ),
            "loaded_models": list(self.loaded_models.keys()),
            "loading_models": list(self.model_loading_in_progress.keys()),
            "waiting_requests": len(self.waiting_requests),
            "running_requests": len(self.running_requests),
            "completed_requests": len(self.completed_requests),
            "utilization": self.utilization,
        }


class NodeMemoryPlanner:
    """
    Memory planner for node-level resource management.

    Manages memory allocation across multiple GPUs in a node,
    considering both GPU memory and node-level constraints.
    """

    def __init__(self, config: NodeConfig):
        """
        Initialize node memory planner.

        Args:
            config: Node configuration
        """
        self.config = config
        self.total_memory_gb = config.total_memory_gb
        self.allocated_memory_gb = 0.0

    def can_allocate_model(self, model_memory_gb: float) -> bool:
        """
        Check if the node has enough memory for a model.

        Args:
            model_memory_gb: Memory required for the model in GB

        Returns:
            bool: True if allocation is possible
        """
        return self.allocated_memory_gb + model_memory_gb <= self.total_memory_gb

    def allocate_model(self, model_memory_gb: float) -> bool:
        """
        Allocate memory for a model.

        Args:
            model_memory_gb: Memory required for the model in GB

        Returns:
            bool: True if allocation was successful
        """
        if self.can_allocate_model(model_memory_gb):
            self.allocated_memory_gb += model_memory_gb
            return True
        return False

    def free_model(self, model_memory_gb: float):
        """
        Free memory previously allocated for a model.

        Args:
            model_memory_gb: Memory to free in GB
        """
        self.allocated_memory_gb = max(0, self.allocated_memory_gb - model_memory_gb)
