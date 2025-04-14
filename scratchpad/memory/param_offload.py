import torch
import threading
import time
import queue
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from threading import Thread
from collections import defaultdict, deque
from dataclasses import dataclass
import weakref

from scratchpad.utils import logger, is_pin_memory_available
from torch import nn


@dataclass
class LayerUsageInfo:
    """Information about layer usage patterns."""

    # Last time this layer was accessed
    last_accessed: float = 0.0
    # Number of times this layer was accessed
    access_count: int = 0
    # Whether this layer is currently on GPU
    on_gpu: bool = False
    # Whether this layer is currently being prefetched
    is_prefetching: bool = False
    # Priority score (higher means more likely to keep in GPU)
    priority_score: float = 0.0


class ModuleState:
    """Represents the state of a module's parameters."""

    def __init__(self, module: nn.Module):
        # Save original parameter states
        self.param_states: Dict[str, Dict] = {}

        # Handle case where module has no parameters
        params = list(module.parameters())
        if not params:
            logger.warning(
                f"Module has no parameters. Using 'cuda:0' as default device."
            )
            self.device = torch.device("cuda:0")
        else:
            self.device = params[0].device

        self.pin_memory = is_pin_memory_available()

        # Save original parameters
        for name, param in module.named_parameters():
            self.param_states[name] = {
                "data": param.data,
                "device": param.data.device,
                "offloaded": False,
            }

    def offload_to_cpu(self, module: nn.Module) -> int:
        """Offload module parameters to CPU and return bytes saved."""
        bytes_saved = 0

        for name, param in module.named_parameters():
            if param.device.type != "cpu":
                # Create empty CPU tensor with pinned memory if available
                cpu_tensor = torch.empty_strided(
                    size=param.data.size(),
                    stride=param.data.stride(),
                    dtype=param.data.dtype,
                    layout=param.data.layout,
                    device="cpu",
                    pin_memory=self.pin_memory,
                )

                # Copy data to CPU
                cpu_tensor.copy_(param.data)

                # Update parameter data
                param.data = cpu_tensor

                # Calculate bytes saved
                bytes_saved += param.data.numel() * param.data.element_size()

                # Update state
                self.param_states[name]["offloaded"] = True

        return bytes_saved

    def load_to_gpu(self, module: nn.Module) -> int:
        """Load module parameters back to GPU and return bytes loaded."""
        bytes_loaded = 0

        for name, param in module.named_parameters():
            if param.device.type == "cpu":
                # Move data back to original device
                param.data = param.data.to(self.device, non_blocking=True)

                # Calculate bytes loaded
                bytes_loaded += param.data.numel() * param.data.element_size()

                # Update state
                self.param_states[name]["offloaded"] = False

        return bytes_loaded


class ParameterOffloadManager:
    """Manager for parameter offloading between CPU and GPU memory."""

    def __init__(
        self,
        enable_offload: bool = False,
        enable_prefetch: bool = True,
        cpu_offload_ratio: float = 0.7,
        prefetch_window: int = 2,
    ):
        self.enable_offload = enable_offload
        self.enable_prefetch = enable_prefetch and enable_offload
        self.cpu_offload_ratio = cpu_offload_ratio
        self.prefetch_window = prefetch_window

        # Store module information
        self.modules: Dict[str, nn.Module] = {}
        self.module_states: Dict[str, ModuleState] = {}
        self.usage_info: Dict[str, LayerUsageInfo] = defaultdict(LayerUsageInfo)
        self.execution_order: List[str] = []
        self.current_layer_idx: int = -1

        # Track offloaded bytes
        self.total_offloaded_bytes: int = 0
        self.total_parameter_bytes: int = 0

        # Prefetching queue and thread
        self.prefetch_queue = queue.Queue()
        self.prefetch_thread = None
        self.stop_prefetch = threading.Event()

        logger.info(
            f"Parameter offloading initialized (enabled={enable_offload}, prefetch={enable_prefetch})"
        )

    def register_module(
        self, name: str, module: nn.Module, priority: float = 1.0
    ) -> None:
        """Register a module for potential CPU offloading."""
        if not self.enable_offload:
            return

        # Skip if module already registered
        if name in self.modules:
            return

        # Register module
        self.modules[name] = module
        self.module_states[name] = ModuleState(module)
        self.usage_info[name].priority_score = priority
        self.execution_order.append(name)

        # Track total parameter bytes
        for param in module.parameters():
            self.total_parameter_bytes += param.numel() * param.element_size()

        logger.info(
            f"Registered module {name} for potential offloading (priority={priority})"
        )

    def register_model_layers(
        self, model: nn.Module, layer_prefix: str = "layers"
    ) -> None:
        """Register all transformer layers in a model."""
        if not self.enable_offload:
            return

        # Find all modules that have transformer layers
        for name, module in model.named_modules():
            if layer_prefix in name and isinstance(module, nn.Module):
                # Extract layer number - assuming format like layers.1, decoder.layers.2, etc.
                parts = name.split(".")
                for i, part in enumerate(parts):
                    if (
                        part == layer_prefix
                        and i + 1 < len(parts)
                        and parts[i + 1].isdigit()
                    ):
                        layer_num = int(parts[i + 1])
                        # Higher layer numbers get higher priority (we typically want to keep early layers in GPU)
                        priority = layer_num + 1
                        self.register_module(name, module, priority=priority)

        # Sort execution order by expected execution order (layer numbers)
        self.execution_order.sort(
            key=lambda x: [int(part) if part.isdigit() else 0 for part in x.split(".")]
        )

        logger.info(
            f"Registered {len(self.modules)} transformer layers for CPU offloading"
        )

    def start_offloading(self) -> None:
        """Start offloading parameters based on configuration."""
        if not self.enable_offload or not self.modules:
            return

        # Calculate how many bytes to offload
        target_offload_bytes = int(self.total_parameter_bytes * self.cpu_offload_ratio)

        # Sort modules by priority (lowest priority first - they'll be offloaded first)
        sorted_modules = sorted(
            self.usage_info.items(), key=lambda x: x[1].priority_score
        )

        # Offload modules until we reach target
        for name, _ in sorted_modules:
            if name in self.modules and name in self.module_states:
                module = self.modules[name]
                module_state = self.module_states[name]

                # Offload and track bytes
                bytes_saved = module_state.offload_to_cpu(module)
                self.total_offloaded_bytes += bytes_saved
                self.usage_info[name].on_gpu = False

                # Stop if we've reached our target
                if self.total_offloaded_bytes >= target_offload_bytes:
                    break

        # Start prefetch thread if enabled
        if self.enable_prefetch:
            self.stop_prefetch.clear()
            self.prefetch_thread = Thread(target=self._prefetch_worker, daemon=True)
            self.prefetch_thread.start()

        logger.info(
            f"Offloaded {self.total_offloaded_bytes/1024/1024:.2f}MB of parameters to CPU "
            f"({self.total_offloaded_bytes/self.total_parameter_bytes*100:.1f}% of total)"
        )

    def stop_offloading(self) -> None:
        """Stop offloading and load all parameters back to GPU."""
        if not self.enable_offload:
            return

        # Stop prefetcher
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            self.stop_prefetch.set()
            self.prefetch_thread.join(timeout=5.0)

        # Load all modules back to GPU
        for name, module in self.modules.items():
            if name in self.module_states:
                self.module_states[name].load_to_gpu(module)
                self.usage_info[name].on_gpu = True

        # Reset counters
        self.total_offloaded_bytes = 0
        self.current_layer_idx = -1

        logger.info("All parameters loaded back to GPU")

    def pre_forward_hook(self, module_name: str) -> None:
        """Hook called before a module's forward pass."""
        if not self.enable_offload or module_name not in self.modules:
            return

        # Update usage info
        self.usage_info[module_name].last_accessed = time.time()
        self.usage_info[module_name].access_count += 1

        # If module is in execution order, update current position
        if module_name in self.execution_order:
            self.current_layer_idx = self.execution_order.index(module_name)

        # Ensure module is on GPU
        if not self.usage_info[module_name].on_gpu:
            module = self.modules[module_name]
            module_state = self.module_states[module_name]
            module_state.load_to_gpu(module)
            self.usage_info[module_name].on_gpu = True

        # Request prefetch of next modules if prefetching is enabled
        if self.enable_prefetch and self.current_layer_idx >= 0:
            self._queue_prefetch()

    def post_forward_hook(self, module_name: str) -> None:
        """Hook called after a module's forward pass."""
        # Currently, we don't offload immediately after use
        # This could be added as an optimization if needed
        pass

    def _queue_prefetch(self) -> None:
        """Queue prefetch requests for upcoming modules."""
        if self.current_layer_idx < 0 or not self.enable_prefetch:
            return

        for i in range(self.prefetch_window):
            next_idx = self.current_layer_idx + 1 + i
            if next_idx < len(self.execution_order):
                next_name = self.execution_order[next_idx]
                if (
                    not self.usage_info[next_name].on_gpu
                    and not self.usage_info[next_name].is_prefetching
                ):
                    self.usage_info[next_name].is_prefetching = True
                    self.prefetch_queue.put(next_name)

    def _prefetch_worker(self) -> None:
        """Worker thread for prefetching parameters."""
        while not self.stop_prefetch.is_set():
            try:
                # Get next module to prefetch (with timeout to allow checking stop flag)
                module_name = self.prefetch_queue.get(timeout=0.1)

                # Skip if module already on GPU or not registered
                if (
                    module_name not in self.modules
                    or self.usage_info[module_name].on_gpu
                ):
                    self.usage_info[module_name].is_prefetching = False
                    self.prefetch_queue.task_done()
                    continue

                # Load module to GPU
                module = self.modules[module_name]
                module_state = self.module_states[module_name]
                module_state.load_to_gpu(module)

                # Update state
                self.usage_info[module_name].on_gpu = True
                self.usage_info[module_name].is_prefetching = False

                # Mark task as done
                self.prefetch_queue.task_done()

            except queue.Empty:
                # Queue empty, just continue
                continue
            except Exception as e:
                logger.error(f"Error in prefetch worker: {e}")
                self.usage_info[module_name].is_prefetching = False
                self.prefetch_queue.task_done()


# Create a global instance for easy access
parameter_offload_manager = None


def init_parameter_offload_manager(
    enable_offload: bool = False,
    enable_prefetch: bool = True,
    cpu_offload_ratio: float = 0.7,
    prefetch_window: int = 2,
) -> ParameterOffloadManager:
    """Initialize the global parameter offload manager."""
    global parameter_offload_manager
    parameter_offload_manager = ParameterOffloadManager(
        enable_offload=enable_offload,
        enable_prefetch=enable_prefetch,
        cpu_offload_ratio=cpu_offload_ratio,
        prefetch_window=prefetch_window,
    )
    return parameter_offload_manager


def get_parameter_offload_manager() -> Optional[ParameterOffloadManager]:
    """Get the global parameter offload manager."""
    return parameter_offload_manager
