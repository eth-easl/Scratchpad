from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Set, Tuple, Union

from humanize import naturalsize

from core.env import calculate_model_loading_time
from core.memory_planner import ParallelConfig
from utils.model_utils import get_model_size_gb

from .engine import LLMEngine
from .request import GenerationRequest
from .trace import TraceEvent

BYTES_PER_GB = 1_000_000_000


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
        self.gpu_configs = {gpu.gpu_id: gpu for gpu in config.gpus}
        self.gpu_precisions: Dict[int, Tuple[int, int, int]] = {}
        self.engine_gpu_groups: Dict[LLMEngine, Set[int]] = {}

        # Model management
        self.loaded_models = {}  # model_name -> Set[gpu_ids]
        self.model_loading_in_progress: Dict[str, Dict[int, float]] = {}

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
        gpu_config: Union[GPUConfig, List[GPUConfig]],
        model_name: str = None,
        preload_time: Optional[float] = None,
        w_bit: int = 4,
        a_bit: int = 4,
        kv_bit: int = 4,
        parallel_config: Optional[ParallelConfig] = None,
    ) -> bool:
        """
        Add one or more GPUs as a single logical engine to the node.

        Args:
            gpu_config: GPU configuration or list of configurations to bind together
            model_name: Model to load on this engine (optional)
            w_bit: Weight precision
            a_bit: Activation precision
            kv_bit: KV cache precision
            parallel_config: Optional parallelism configuration override

        Returns:
            bool: True if GPUs were added successfully
        """
        if isinstance(gpu_config, (list, tuple, set)):
            gpu_group = list(gpu_config)
        else:
            gpu_group = [gpu_config]

        if not gpu_group:
            return False

        gpu_ids = [gpu.gpu_id for gpu in gpu_group]
        if not set(gpu_ids).issubset(self.available_gpus):
            return False

        hardware_names = {gpu.hardware_name for gpu in gpu_group}
        if len(hardware_names) != 1:
            raise ValueError("All GPUs in a tensor-parallel group must share hardware.")

        memory_gb = getattr(gpu_group[0], "memory_gb", None)
        memory_override_bytes = (
            int(memory_gb * BYTES_PER_GB) if memory_gb is not None else None
        )

        pipeline_parallel_size = 1
        if isinstance(parallel_config, ParallelConfig):
            pipeline_parallel_size = parallel_config.pipeline_parallel_size
        elif isinstance(parallel_config, dict):
            pipeline_parallel_size = int(
                parallel_config.get("pipeline_parallel_size", 1)
            )

        effective_parallel_config = ParallelConfig(
            tensor_parallel_size=max(1, len(gpu_group)),
            pipeline_parallel_size=max(1, pipeline_parallel_size),
        )

        engine_id = f"{self.node_id}-{'_'.join(str(gid) for gid in sorted(gpu_ids))}"
        engine = LLMEngine(
            engine_id,
            model_name or "placeholder",
            gpu_group[0].hardware_name,
            w_bit,
            a_bit,
            kv_bit,
            parallel_config=effective_parallel_config,
            memory_override_bytes=memory_override_bytes,
        )

        for gpu in gpu_group:
            self.gpus[gpu.gpu_id] = engine
            self.available_gpus.discard(gpu.gpu_id)
            self.gpu_memory_usage[gpu.gpu_id] = 0.0
            self.gpu_precisions[gpu.gpu_id] = (w_bit, a_bit, kv_bit)
            self.gpu_configs[gpu.gpu_id] = gpu

        self.engine_gpu_groups[engine] = set(gpu_ids)

        if model_name:
            for gpu in gpu_group:
                self.load_model_on_gpu(
                    model_name,
                    gpu.gpu_id,
                    preload_time=preload_time,
                )

        return True

    def expand_engine(
        self,
        engine: LLMEngine,
        new_gpu_configs: List[GPUConfig],
        *,
        preload_time: Optional[float] = None,
    ) -> bool:
        """
        Attach additional GPUs to an existing engine, increasing tensor parallelism.

        Args:
            engine: Engine instance to expand
            new_gpu_configs: GPUs to attach to the engine
            preload_time: Optional preload duration to reuse for new GPUs

        Returns:
            bool: True if expansion succeeded
        """
        if engine not in self.engine_gpu_groups:
            return False

        new_gpu_configs = list(new_gpu_configs)
        if not new_gpu_configs:
            return False

        # Ensure all GPUs are available and compatible
        for gpu in new_gpu_configs:
            if gpu.gpu_id not in self.available_gpus:
                return False
            if gpu.hardware_name != engine.hardware_name:
                raise ValueError(
                    "All GPUs in a tensor-parallel group must share hardware."
                )

        # Attach GPUs to engine
        for gpu in new_gpu_configs:
            self.gpus[gpu.gpu_id] = engine
            self.available_gpus.discard(gpu.gpu_id)
            self.gpu_memory_usage[gpu.gpu_id] = 0.0
            self.gpu_precisions[gpu.gpu_id] = (
                engine.w_bit,
                engine.a_bit,
                engine.kv_bit,
            )
            self.gpu_configs[gpu.gpu_id] = gpu
            self.engine_gpu_groups[engine].add(gpu.gpu_id)

        # Update engine parallel configuration
        new_parallel_config = ParallelConfig(
            tensor_parallel_size=len(self.engine_gpu_groups[engine]),
            pipeline_parallel_size=engine.parallel_config.pipeline_parallel_size,
        )
        engine.update_parallel_config(new_parallel_config)

        model_name = engine.model_name
        if model_name and model_name != "placeholder":
            for gpu in new_gpu_configs:
                self.load_model_on_gpu(
                    model_name,
                    gpu.gpu_id,
                    preload_time=preload_time,
                )

        return True

    def load_model_on_gpu(
        self,
        model_name: str,
        gpu_id: int,
        preload_time: Optional[float] = None,
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
        if preload_time is None:
            weight_bits, _, _ = self.gpu_precisions.get(gpu_id, (16, 16, 16))
            model_size_gb = get_model_size_gb(
                model_name,
                weight_bits=weight_bits,
            )
            loading_time = calculate_model_loading_time(
                model_size_gb=model_size_gb,
                disk_bandwidth_mbps=self.config.disk_bandwidth_mbps,
                pcie_bandwidth_gbps=self.config.pcie_bandwidth_gbps,
                cache_available=False,
            )
        else:
            loading_time = max(preload_time, 0.0)

        completion_time = self.current_time + loading_time

        # Track model loading
        if model_name not in self.model_loading_in_progress:
            self.model_loading_in_progress[model_name] = {}

        self.model_loading_in_progress[model_name][gpu_id] = completion_time
        self.next_available_time = max(self.next_available_time, completion_time)

        return completion_time

    def begin_model_switch(
        self, target_model: str, current_time: float, loading_duration: float
    ) -> None:
        """Prepare this node to load a new model on all GPUs."""
        self.current_time = current_time
        self.loaded_models.clear()
        self.model_loading_in_progress.clear()
        self.waiting_requests.clear()
        self.running_requests.clear()

        loading_duration = max(loading_duration, 0.0)

        for engine in self.gpus.values():
            engine.waiting.clear()
            engine.running.clear()

        for gpu_id in self.gpus.keys():
            self.load_model_on_gpu(
                target_model,
                gpu_id,
                preload_time=loading_duration,
            )
        self.next_available_time = max(
            self.next_available_time, current_time + max(loading_duration, 0.0)
        )

    def _ensure_engine_capacity(
        self,
        engine: LLMEngine,
        model_name: str,
        *,
        preload_time: Optional[float],
    ) -> None:
        """
        Ensure an engine has sufficient GPUs allocated to serve the model.

        Expands tensor parallel groups by borrowing available GPUs of the same
        hardware type when needed.
        """
        if engine.memory_planner.get_max_num_blocks() > 0:
            return

        available_same_hw = [
            self.gpu_configs[gpu_id]
            for gpu_id in sorted(self.available_gpus)
            if self.gpu_configs[gpu_id].hardware_name == engine.hardware_name
        ]

        while engine.memory_planner.get_max_num_blocks() == 0 and available_same_hw:
            next_gpu = available_same_hw.pop(0)
            expanded = self.expand_engine(
                engine,
                [next_gpu],
                preload_time=preload_time,
            )
            if not expanded:
                break

        if engine.memory_planner.get_max_num_blocks() == 0:
            per_shard = engine.memory_planner.get_weights_memory() / max(
                engine.parallel_config.tensor_parallel_size, 1
            )
            group_gpu_ids = self.engine_gpu_groups.get(engine, set())
            if group_gpu_ids:
                sample_gpu = next(iter(group_gpu_ids))
                gpu_config = self.gpu_configs.get(sample_gpu)
                memory_bytes = gpu_config.memory_gb * BYTES_PER_GB if gpu_config else 0
            else:
                memory_bytes = 0
            print(
                f"Warning: Model weights per shard ({naturalsize(per_shard)}) exceed GPU memory "
                f"({naturalsize(memory_bytes)}). Unable to allocate additional GPUs for {model_name} "
                f"on node {self.node_id}."
            )

    def finalize_model_loading(self, model_name: str, gpu_ids: Set[int]):
        """Finalize GPU state once model loading completes."""

        if model_name not in self.loaded_models:
            self.loaded_models[model_name] = set()

        self.loaded_models[model_name].update(gpu_ids)

        for gpu_id in gpu_ids:
            engine = self.gpus.get(gpu_id)
            if not engine:
                continue

            w_bit, a_bit, kv_bit = self.gpu_precisions.get(
                gpu_id, (engine.w_bit, engine.a_bit, engine.kv_bit)
            )

            if engine.model_name != model_name:
                engine.reconfigure_model(
                    model_name,
                    w_bit=w_bit,
                    a_bit=a_bit,
                    kv_bit=kv_bit,
                )

            self._ensure_engine_capacity(
                engine,
                model_name,
                preload_time=None,
            )

    def check_model_loading_completions(self):
        """Check for completed model loading and update GPU availability."""
        completed_models = []

        for model_name, gpu_completion in list(self.model_loading_in_progress.items()):
            completed_gpu_ids = {
                gpu_id
                for gpu_id, end_time in list(gpu_completion.items())
                if self.current_time >= end_time
            }

            if completed_gpu_ids:
                self.finalize_model_loading(model_name, completed_gpu_ids)
                for gpu_id in completed_gpu_ids:
                    gpu_completion.pop(gpu_id, None)

            if not gpu_completion:
                completed_models.append(model_name)

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
        seen_engines: Set[LLMEngine] = set()
        loaded_gpu_ids = self.loaded_models[model_name]

        for gpu_id in sorted(loaded_gpu_ids):
            engine = self.gpus.get(gpu_id)
            if engine is None or engine in seen_engines:
                continue
            group_gpu_ids = self.engine_gpu_groups.get(engine, set())
            if group_gpu_ids and not group_gpu_ids.issubset(loaded_gpu_ids):
                continue
            if len(engine.running) == 0:  # Engine group is not busy
                available_gpus.append(gpu_id)
                seen_engines.add(engine)

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
        processed_engines: Set[LLMEngine] = set()
        for _, engine in self.gpus.items():
            if engine in processed_engines:
                continue
            processed_engines.add(engine)
            engine_event, finished_reqs, next_time, memory_event = engine.step(
                current_time
            )

            if engine_event is not None:
                events.extend(engine_event)
                if memory_event is not None:
                    events.append(memory_event)

            if finished_reqs:
                completed_requests.extend(finished_reqs)
                # Move finished requests from running to completed
                for finished_req in finished_reqs:
                    if finished_req in self.running_requests:
                        self.running_requests.remove(finished_req)
                        self.completed_requests.append(finished_req)

            if next_time is not None:
                next_times.append(next_time)

        # Determine next time step; None indicates the node can sleep until new work arrives
        next_time = min(next_times) if next_times else None
        if next_times:
            self.next_available_time = max(self.next_available_time, max(next_times))
        else:
            self.next_available_time = max(self.next_available_time, current_time)

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

    @property
    def is_idle(self) -> bool:
        """Check if the node is idle (no running requests and no loading in progress)."""
        return (
            len(self.running_requests) == 0 and len(self.model_loading_in_progress) == 0
        )

    @property
    def utilization(self) -> float:
        """Calculate node utilization (0.0 to 1.0)."""
        if not self.gpu_configs:
            return 0.0

        total_gpu_slots = len(self.gpu_configs)
        busy_gpu_slots = 0
        for engine, gpu_ids in self.engine_gpu_groups.items():
            if len(engine.running) > 0:
                busy_gpu_slots += len(gpu_ids)

        return busy_gpu_slots / total_gpu_slots if total_gpu_slots else 0.0

    def get_status(self) -> Dict:
        """Get current node status for monitoring."""
        busy_gpu_slots = 0
        for engine, gpu_ids in self.engine_gpu_groups.items():
            if len(engine.running) > 0:
                busy_gpu_slots += len(gpu_ids)

        return {
            "node_id": self.node_id,
            "name": self.name,
            "total_gpus": len(self.gpu_configs),
            "busy_gpus": busy_gpu_slots,
            "loaded_models": list(self.loaded_models.keys()),
            "loading_models": list(self.model_loading_in_progress.keys()),
            "waiting_requests": len(self.waiting_requests),
            "running_requests": len(self.running_requests),
            "completed_requests": len(self.completed_requests),
            "utilization": self.utilization,
        }

    def collect_engine_status(self) -> List[Dict[str, Any]]:
        """Return per-GPU engine status snapshots for periodic logging."""
        snapshots: List[Dict[str, Any]] = []
        for engine, gpu_ids in self.engine_gpu_groups.items():
            engine_status = engine.status_snapshot()
            engine_status.update(
                {
                    "node_id": self.node_id,
                    "gpu_ids": sorted(gpu_ids),
                }
            )
            snapshots.append(engine_status)
        return snapshots


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
