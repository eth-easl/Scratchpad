import heapq
import json
import math
import time
from collections import defaultdict, deque
from itertools import count
from typing import Any, Dict, List, Optional, Set
from .node import ComputeNode, NodeConfig, GPUConfig
from .request import GenerationRequest
from .trace import TraceEvent
from .policies.routing import RandomNodePolicy
from .policies.node_reprovisioning import (
    NodeReprovisioningPolicy,
    RandomNodeReprovisioningPolicy,
)
from core.env import EnvironmentConfig, calculate_model_loading_time
from utils.model_utils import BYTES_PER_GB, get_model_size_gb


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
        self.pending_requests: List[GenerationRequest] = []
        self._ready_requests: deque[GenerationRequest] = deque()
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
        self.node_reprovisioning_in_progress = (
            {}
        )  # node_id -> (end_time, target_model, previous_models)

        # Event-driven scheduler state
        self._node_event_queue: List[tuple[float, int, ComputeNode]] = []
        self._node_event_counter = count()
        self._node_timers: Dict[int, Optional[float]] = {}

        # Progress tracking for timeout
        self.last_progress_time = 0.0
        self.simulation_timeout = 300.0  # 5 minutes timeout
        self.print_interval = max(0.0, print_interval)
        self.last_print_time = 0.0
        self.last_print_wall_time = time.monotonic()

        # Initialize nodes from environment configuration
        if environment_config:
            self._initialize_nodes_from_config(environment_config)

    def _record_instant_event(
        self,
        name: str,
        category: str,
        timestamp_s: float,
        *,
        pid: str | int = "system",
        tid: str | int = 0,
        args: Optional[dict] = None,
    ) -> None:
        """Capture an instant-style trace event for human-readable traces."""
        self._trace.append(
            TraceEvent(
                name=name,
                cat=category,
                ph="X",
                pid=pid,
                tid=tid,
                ts=int(max(timestamp_s, 0.0) * 1_000_000),
                dur=0,
                args=args or {},
            )
        )

    def record_assignment_event(
        self,
        request: GenerationRequest,
        node: ComputeNode,
        assigned_at: float,
    ) -> None:
        """Log when a request is routed to a node."""
        self._record_instant_event(
            name=f"Assigned :: request {request.req_id}",
            category="request.assignment",
            timestamp_s=assigned_at,
            pid=f"node-{node.node_id}",
            args={
                "request_id": request.req_id,
                "model": request.model,
                "node_id": node.node_id,
                "node_name": node.name,
                "arrival_time_s": round(request.arrive_at, 6),
                "queued_duration_s": round(
                    max(assigned_at - request.arrive_at, 0.0), 6
                ),
            },
        )

    def record_completion_event(
        self,
        request: GenerationRequest,
        node: ComputeNode,
    ) -> None:
        """Log when a request finishes executing on a node."""
        finished_at = request.generation_finished_at or self.global_timer
        args = {
            "request_id": request.req_id,
            "model": request.model,
            "node_id": node.node_id,
            "node_name": node.name,
            "arrival_time_s": round(request.arrive_at, 6),
            "completed_at_s": round(finished_at, 6),
            "latency_s": round(finished_at - request.arrive_at, 6),
            "output_tokens_generated": request.generated_tokens,
        }
        if request.prefill_finished_at is not None:
            args["prefill_finished_at_s"] = round(request.prefill_finished_at, 6)
        self._record_instant_event(
            name=f"Completed :: request {request.req_id}",
            category="request.lifecycle",
            timestamp_s=finished_at,
            pid=f"node-{node.node_id}",
            args=args,
        )

    def _reset_node_event_queue(self) -> None:
        """Reset node event scheduling structures."""
        self._node_event_queue.clear()
        self._node_event_counter = count()

    def _schedule_node(self, node: ComputeNode, ready_time: float) -> None:
        """
        Schedule a node for processing at the specified simulation time.

        Args:
            node: ComputeNode instance that has pending work
            ready_time: Simulation timestamp when the node should be revisited
        """
        self._node_timers[node.node_id] = ready_time
        heapq.heappush(
            self._node_event_queue,
            (ready_time, next(self._node_event_counter), node),
        )

    def _process_node_events(self) -> bool:
        """
        Execute node steps for all events scheduled at the current simulation time.

        Returns:
            bool: True if any progress was made at the current simulation time
        """
        progress_made = False
        while (
            self._node_event_queue and self._node_event_queue[0][0] <= self.global_timer
        ):
            event_time, _, node = heapq.heappop(self._node_event_queue)
            scheduled_time = self._node_timers.get(node.node_id)

            if scheduled_time is None or abs(event_time - scheduled_time) > 1e-9:
                continue

            if event_time > self.global_timer:
                self.global_timer = event_time

            node_events, completed_requests, next_time = node.step(event_time)
            progress_made = True

            if node_events:
                self._trace.extend(node_events)

            if completed_requests:
                for finished_request in completed_requests:
                    self.record_completion_event(finished_request, node)

            if next_time is not None:
                self._schedule_node(node, next_time)
            else:
                self._node_timers[node.node_id] = None

        return progress_made

    def _next_pending_request_arrival(self, after_time: float) -> Optional[float]:
        """Return the earliest pending request arrival after the provided time."""
        if not self.pending_requests:
            return None

        next_request = self.pending_requests[0]
        if next_request.arrive_at > after_time:
            return next_request.arrive_at

        # In rare cases when pending_requests was not fully drained for current time
        for req in self.pending_requests:
            if req.arrive_at > after_time:
                return req.arrive_at
        return None

    def _next_node_reprovision_completion_time(
        self, after_time: float
    ) -> Optional[float]:
        """Return the next scheduled node re-provision completion time."""
        next_time: Optional[float] = None
        for end_time, _, _ in self.node_reprovisioning_in_progress.values():
            if end_time > after_time:
                if next_time is None or end_time < next_time:
                    next_time = end_time
        return next_time

    def on_node_request_assigned(
        self, node: ComputeNode, *, scheduled_time: Optional[float] = None
    ) -> None:
        """
        Notify the scheduler that a node has new work to process.

        Args:
            node: Node that received new requests
            scheduled_time: Optional override for when processing should occur
        """
        ready_time = scheduled_time if scheduled_time is not None else self.global_timer
        self._schedule_node(node, ready_time)

    def _collect_status_snapshot(self) -> Dict[str, Any]:
        """Aggregate engine and node status for periodic logging."""
        engine_snapshots: List[Dict[str, Any]] = []
        total_blocks = 0
        used_blocks = 0
        node_block_usage: Dict[int, Dict[str, int]] = defaultdict(
            lambda: {"used": 0, "total": 0}
        )
        busy = idle = queued = 0

        for node in self.nodes.values():
            for snapshot in node.collect_engine_status():
                engine_snapshots.append(snapshot)
                total_blocks += snapshot.get("total_blocks", 0)
                used_blocks += snapshot.get("used_blocks", 0)
                node_id = snapshot.get("node_id")
                if node_id is not None:
                    node_block_usage[node_id]["used"] += snapshot.get("used_blocks", 0)
                    node_block_usage[node_id]["total"] += snapshot.get(
                        "total_blocks", 0
                    )
                state = snapshot.get("state", "idle")
                if state == "busy":
                    busy += 1
                elif state == "queued":
                    queued += 1
                else:
                    idle += 1

        reprovisioning = []
        for node_id, (
            end_time,
            target_model,
            previous_models,
        ) in self.node_reprovisioning_in_progress.items():
            reprovisioning.append(
                {
                    "node_id": node_id,
                    "target_model": target_model,
                    "eta": max(end_time - self.global_timer, 0.0),
                }
            )

        pending_count = len(self.pending_requests) + len(self._ready_requests)
        completed_requests = sum(
            len(node.completed_requests) for node in self.nodes.values()
        )

        return {
            "timestamp": self.global_timer,
            "total_engines": len(engine_snapshots),
            "busy_engines": busy,
            "queued_engines": queued,
            "idle_engines": idle,
            "used_blocks": used_blocks,
            "total_blocks": total_blocks,
            "pending_requests": pending_count,
            "completed_requests": completed_requests,
            "total_requests": self.total_requests,
            "finished_pct": self.finished_percentage,
            "reprovisioning": reprovisioning,
            "node_block_usage": node_block_usage,
        }

    def _log_status(self) -> None:
        """Print a periodic, human-readable status line."""
        snapshot = self._collect_status_snapshot()
        total_blocks = snapshot["total_blocks"]
        used_blocks = snapshot["used_blocks"]
        if total_blocks > 0:
            block_msg = f"blocks {used_blocks}/{total_blocks}"
        else:
            block_msg = "blocks 0/0"

        total_engines = snapshot["total_engines"]
        busy = snapshot["busy_engines"]
        queued = snapshot["queued_engines"]
        idle = snapshot["idle_engines"]
        engines_msg = (
            f"engines busy/queued/idle {busy}/{queued}/{idle} (total {total_engines})"
        )

        reprovisioning = snapshot["reprovisioning"]
        if reprovisioning:
            reprovision_msg = ", ".join(
                [
                    f"node {item['node_id']}→{item['target_model']} (eta {item['eta']:.2f}s)"
                    for item in reprovisioning
                ]
            )
        else:
            reprovision_msg = "none"

        node_block_usage = snapshot["node_block_usage"]
        if node_block_usage:
            per_node_blocks = ", ".join(
                [
                    f"node {node_id}:{usage['used']}/{usage['total']}"
                    for node_id, usage in sorted(node_block_usage.items())
                ]
            )
        else:
            per_node_blocks = "none"

        print(
            " | ".join(
                [
                    f"T={snapshot['timestamp']:.2f}s",
                    f"requests {snapshot['completed_requests']}/{snapshot['total_requests']}"
                    f" ({snapshot['finished_pct']:.2f}%)",
                    block_msg,
                    f"node blocks {per_node_blocks}",
                    engines_msg,
                    f"pending requests {snapshot['pending_requests']}",
                    f"reprovisioning: {reprovision_msg}",
                ]
            )
        )

    def record_reprovision_completion(
        self,
        node: ComputeNode,
        new_model: str,
        current_time: float,
        previous_models: Optional[List[str]] = None,
    ) -> None:
        """Log when a node finishes re-provisioning."""
        self._record_instant_event(
            name=f"Node {node.node_id} ready :: {new_model}",
            category="node.reprovisioning",
            timestamp_s=current_time,
            pid=f"node-{node.node_id}",
            args={
                "node_id": node.node_id,
                "node_name": node.name,
                "new_model": new_model,
                "previous_models": previous_models or [],
                "reprovision_completed_at_s": round(current_time, 6),
            },
        )

    def _initialize_nodes_from_config(self, config: EnvironmentConfig):
        """
        Initialize nodes from environment configuration.

        Args:
            config: Environment configuration containing node settings
        """
        print(f"Initializing nodes from environment configuration...")

        total_loading_time = 0.0

        # Prefer new node-based configuration over legacy GPU configuration
        if config.nodes:
            print(f"Using node-based configuration with {len(config.nodes)} nodes")
            for node_config_data in config.nodes:
                # Group GPUs by hardware type to allow tensor-parallel placement
                gpus_by_type: Dict[str, List[GPUConfig]] = defaultdict(list)
                gpu_counter = 0
                for gpu_type, gpu_count, memory_gb in zip(
                    node_config_data.gpu_types,
                    node_config_data.gpu_count_per_type,
                    node_config_data.memory_per_gpu_gb,
                ):
                    for _ in range(gpu_count):
                        gpu = GPUConfig(
                            gpu_id=gpu_counter,
                            hardware_name=gpu_type,
                            memory_gb=memory_gb,
                        )
                        gpus_by_type[gpu_type].append(gpu)
                        gpu_counter += 1

                node_config = NodeConfig(
                    node_id=node_config_data.node_id,
                    name=node_config_data.name,
                    gpus=[gpu for group in gpus_by_type.values() for gpu in group],
                    total_memory_gb=sum(
                        gpu.memory_gb
                        for group in gpus_by_type.values()
                        for gpu in group
                    )
                    * 1.2,  # 20% overhead
                    network_bandwidth_gbps=node_config_data.network_bandwidth_gbps,
                    disk_bandwidth_mbps=config.infrastructure.disk_bandwidth_mbps,
                    pcie_bandwidth_gbps=config.infrastructure.pcie_bandwidth_gbps,
                )

                node = ComputeNode(node_config)

                # For now, assume a single model per node
                model_to_load = "meta-llama/Meta-Llama-3-8B-Instruct"
                model_size_gb = get_model_size_gb(model_to_load)

                engines_created = 0
                for gpu_type, gpu_list in gpus_by_type.items():
                    if not gpu_list:
                        continue

                    per_gpu_memory_bytes = int(
                        getattr(gpu_list[0], "memory_gb", 0) * BYTES_PER_GB
                    )
                    per_gpu_memory_bytes = max(per_gpu_memory_bytes, 1)
                    required_tp = max(
                        1,
                        math.ceil(
                            (model_size_gb * BYTES_PER_GB) / per_gpu_memory_bytes
                        ),
                    )

                    available_gpu_configs = list(gpu_list)
                    while len(available_gpu_configs) >= required_tp:
                        gpu_group = available_gpu_configs[:required_tp]
                        del available_gpu_configs[:required_tp]

                        gpu_ids = [gpu.gpu_id for gpu in gpu_group]

                        if config.model_loading.preload_models:
                            loading_time = 0.0
                            completion_time = 0.0
                            loading_event = TraceEvent(
                                name=f"Node {node_config_data.node_id} preloaded :: {model_to_load}",
                                cat="system.model_loading",
                                ph="X",
                                pid=-node_config_data.node_id,
                                tid=gpu_ids[0],
                                ts=0,
                                dur=0,
                                args={
                                    "model": model_to_load,
                                    "node_id": node_config_data.node_id,
                                    "hardware": gpu_type,
                                    "gpu_ids": gpu_ids,
                                    "tensor_parallel_size": len(gpu_group),
                                    "preloaded": True,
                                },
                            )
                            self._trace.append(loading_event)
                        else:
                            loading_time = calculate_model_loading_time(
                                model_size_gb=model_size_gb,
                                disk_bandwidth_mbps=config.infrastructure.disk_bandwidth_mbps,
                                pcie_bandwidth_gbps=config.infrastructure.pcie_bandwidth_gbps,
                                cache_available=False,
                            )
                            completion_time = loading_time
                            total_loading_time = max(total_loading_time, loading_time)

                            loading_event = TraceEvent(
                                name=f"Node {node_config_data.node_id} load :: {model_to_load}",
                                cat="system.model_loading",
                                ph="X",
                                pid=-node_config_data.node_id,
                                tid=gpu_ids[0],
                                ts=0,
                                dur=int(loading_time * 1_000_000),
                                args={
                                    "model": model_to_load,
                                    "node_id": node_config_data.node_id,
                                    "hardware": gpu_type,
                                    "gpu_ids": gpu_ids,
                                    "tensor_parallel_size": len(gpu_group),
                                    "model_size_gb": model_size_gb,
                                    "loading_time_s": loading_time,
                                },
                            )
                            self._trace.append(loading_event)

                        added = node.add_gpu(
                            gpu_group,
                            model_to_load,
                            completion_time,
                            w_bit=16,
                            a_bit=16,
                            kv_bit=16,
                        )
                        if added:
                            engine = node.gpus[gpu_ids[0]]
                            event_args = loading_event.args if loading_event else None

                            while (
                                engine.memory_planner.get_max_num_blocks() == 0
                                and available_gpu_configs
                            ):
                                extra_gpu = available_gpu_configs[0]
                                expanded = node.expand_engine(
                                    engine,
                                    [extra_gpu],
                                    preload_time=completion_time,
                                )
                                if not expanded:
                                    break
                                available_gpu_configs.pop(0)
                                if event_args is not None:
                                    event_args["gpu_ids"] = sorted(
                                        node.engine_gpu_groups[engine]
                                    )
                                    event_args["tensor_parallel_size"] = len(
                                        node.engine_gpu_groups[engine]
                                    )

                            if engine.memory_planner.get_max_num_blocks() == 0:
                                print(
                                    f"  Warning: Engine {engine.engine_id} for {model_to_load} "
                                    "still has zero KV capacity after tensor parallel expansion. "
                                    "Requests for this model may stall."
                                )

                            engines_created += 1
                        else:
                            print(
                                f"  Warning: Failed to allocate GPU group {gpu_ids} "
                                f"on node {node_config_data.node_id} for {model_to_load}."
                            )

                    if available_gpu_configs:
                        print(
                            f"  Warning: Node {node_config_data.node_id} has "
                            f"{len(available_gpu_configs)} unassigned {gpu_type} GPUs "
                            f"for {model_to_load}; requires tensor_parallel_size "
                            f"{required_tp}."
                        )

                self.nodes[node_config_data.node_id] = node
                self._node_timers[node.node_id] = None
                if engines_created > 0:
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
                model_size_gb = get_model_size_gb(
                    gpu_config.model,
                    weight_bits=gpu_config.precision.weight_bits,
                )

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
                        name=f"Node {node_id} load :: {gpu_config.model}",
                        cat="system.model_loading",
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
                node.add_gpu(
                    gpu,
                    gpu_config.model,
                    completion_time,
                    w_bit=gpu_config.precision.weight_bits,
                    a_bit=gpu_config.precision.activation_bits,
                    kv_bit=gpu_config.precision.kv_bits,
                )

                self.nodes[node_id] = node
                self._node_timers[node.node_id] = None
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
                node.add_gpu(
                    gpu_config,
                    model_name,
                    loading_time,
                    w_bit=16,
                    a_bit=16,
                    kv_bit=16,
                )

            self.supported_models.add(model_name)
            self.nodes_by_model[model_name].append(node)

        self.nodes[node_id] = node
        self._node_timers[node.node_id] = None
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
            self._record_instant_event(
                name=f"Request arrived :: {req.req_id}",
                category="request.lifecycle",
                timestamp_s=req.arrive_at,
                args={
                    "request_id": req.req_id,
                    "model": req.model,
                    "input_tokens": req.input_length,
                    "target_output_tokens": req.output_length,
                },
            )

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
        pending_list = list(self.pending_requests)
        pending_list.sort(key=lambda req: req.arrive_at)
        self.pending_requests = deque(pending_list)
        self._ready_requests.clear()
        self._reset_node_event_queue()
        previous_finished = 0
        self.last_progress_time = self.global_timer
        self.last_print_time = self.global_timer
        self.last_print_wall_time = time.monotonic()

        while True:
            progress_at_current_time = False

            while True:
                loop_progress = False

                assigned_now = self.assign_requests_to_nodes(self.global_timer)
                if assigned_now:
                    loop_progress = True

                if self._process_node_events():
                    loop_progress = True

                if not loop_progress:
                    break

                progress_at_current_time = True

            current_finished = sum(
                len(node.completed_requests) for node in self.nodes.values()
            )
            if current_finished > previous_finished:
                previous_finished = current_finished
                progress_at_current_time = True

            if progress_at_current_time:
                self.last_progress_time = self.global_timer
            elif self.global_timer - self.last_progress_time > self.simulation_timeout:
                print(
                    f"\nSimulation timeout: No progress for {self.simulation_timeout}s"
                )
                print(f"Finished: {self.finished_percentage:.2f}% of requests")
                break

            if self.print_interval > 0.0:
                now_wall = time.monotonic()
                wall_elapsed = now_wall - self.last_print_wall_time
                sim_elapsed = self.global_timer - self.last_print_time
                if (
                    wall_elapsed >= self.print_interval
                    or sim_elapsed >= self.print_interval
                ):
                    self._log_status()
                    self.last_print_time = self.global_timer
                    self.last_print_wall_time = now_wall

            if not self.has_remaining_requests():
                break

            candidate_times: List[float] = []
            if self._node_event_queue:
                candidate_times.append(self._node_event_queue[0][0])
            next_request_arrival = self._next_pending_request_arrival(self.global_timer)
            if next_request_arrival is not None:
                candidate_times.append(next_request_arrival)
            next_reprovision = self._next_node_reprovision_completion_time(
                self.global_timer
            )
            if next_reprovision is not None:
                candidate_times.append(next_reprovision)

            if not candidate_times:
                break

            next_time = min(candidate_times)
            if next_time <= self.global_timer:
                next_time = self.global_timer + 1e-9

            self.global_timer = next_time

        print(f"\nSimulation completed at time: {self.global_timer:.2f}")

    def assign_requests_to_nodes(self, current_time: float) -> int:
        """
        Assign pending requests to available nodes.

        Args:
            current_time: Current simulation time

        Returns:
            int: Number of requests assigned during this call
        """
        assigned_count = 0

        # Check if any node re-provisioning has completed
        completed_reprovisioning = []
        for node_id, (
            end_time,
            target_model,
            previous_models,
        ) in list(self.node_reprovisioning_in_progress.items()):
            if current_time >= end_time and node_id in self.nodes:
                node = self.nodes[node_id]
                node.current_time = current_time
                node.check_model_loading_completions()

                if node not in self.nodes_by_model[target_model]:
                    self.nodes_by_model[target_model].append(node)
                self.supported_models.add(target_model)

                self.record_reprovision_completion(
                    node,
                    target_model,
                    current_time,
                    previous_models,
                )
                completed_reprovisioning.append(node_id)

        # Remove completed re-provisioning from tracking
        for node_id in completed_reprovisioning:
            del self.node_reprovisioning_in_progress[node_id]

        # Move newly arrived requests into the ready queue
        while (
            self.pending_requests and self.pending_requests[0].arrive_at <= current_time
        ):
            self._ready_requests.append(self.pending_requests.popleft())

        if not self._ready_requests:
            return assigned_count

        ready_queue_length = len(self._ready_requests)
        for _ in range(ready_queue_length):
            request = self._ready_requests.popleft()
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

            assigned = False
            if nodes_with_capacity:
                # Use routing policy to select node
                selected_node = self.routing_policy.select_node(
                    request, nodes_with_capacity
                )

                if selected_node and selected_node.add_request(request):
                    self.record_assignment_event(request, selected_node, current_time)
                    self.on_node_request_assigned(
                        selected_node, scheduled_time=current_time
                    )
                    assigned_count += 1
                    assigned = True

            if assigned:
                continue

            # No nodes can serve this model immediately; attempt to re-provision
            nodes_being_reprovisioned = set(self.node_reprovisioning_in_progress.keys())
            selected_node = self.node_reprovisioning_policy.select_node_to_repurpose(
                request.model,
                self.nodes,
                current_time,
                nodes_being_reprovisioned,
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

                removed_models = self._detach_node_from_models(selected_node)
                previous_models = current_models or removed_models

                selected_node.begin_model_switch(
                    request.model, current_time, reprovisioning_time
                )

                # Start re-provisioning
                end_time = current_time + reprovisioning_time
                self.node_reprovisioning_in_progress[selected_node.node_id] = (
                    end_time,
                    request.model,
                    previous_models,
                )
                self._node_timers[selected_node.node_id] = None

                # Add re-provisioning trace event
                reprovision_event = TraceEvent(
                    name=f"Node {selected_node.node_id} reprovision :: {request.model}",
                    cat="node.reprovisioning",
                    ph="X",
                    pid=f"node-{selected_node.node_id}",
                    tid=0,
                    ts=int(current_time * 1_000_000),
                    dur=int(reprovisioning_time * 1_000_000),
                    args={
                        "node_id": selected_node.node_id,
                        "node_name": selected_node.name,
                        "source_models": current_models,
                        "target_model": request.model,
                        "reprovisioning_time_s": reprovisioning_time,
                    },
                )
                self._trace.append(reprovision_event)

            # Request remains queued awaiting assignment
            self._ready_requests.append(request)

        return assigned_count

    def has_remaining_requests(self) -> bool:
        """
        Check if there are any requests remaining to be processed.

        Returns:
            bool: True if there are pending or running requests
        """
        if self.pending_requests or self._ready_requests:
            return True

        # Check if any node re-provisioning is in progress
        if self.node_reprovisioning_in_progress:
            return True

        for node in self.nodes.values():
            if node.waiting_requests or node.running_requests:
                return True

        return False

    def _detach_node_from_models(self, node: ComputeNode) -> List[str]:
        """Remove a node from all routing tables and return prior models."""
        removed_models = []
        for model, nodes_list in self.nodes_by_model.items():
            if node in nodes_list:
                nodes_list.remove(node)
                removed_models.append(model)
                if not nodes_list and model in self.supported_models:
                    self.supported_models.discard(model)
        return removed_models

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
