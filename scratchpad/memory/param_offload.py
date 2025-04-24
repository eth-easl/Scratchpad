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

        # Track module size
        self.module_size_bytes = 0

        # Save original parameter metadata (but not the actual tensor data)
        for name, param in module.named_parameters():
            param_size = param.data.numel() * param.data.element_size()
            self.module_size_bytes += param_size
            self.param_states[name] = {
                "device": param.data.device,  # Only store device info, not the tensor itself
                "shape": param.data.shape,  # Store shape for debugging
                "dtype": param.data.dtype,  # Store dtype for debugging
                "offloaded": False,
                "size_bytes": param_size,  # Store size for memory management
            }
            # We don't store param.data itself to avoid reference leaks

    def get_size_bytes(self) -> int:
        """Get the total size of module parameters in bytes."""
        return self.module_size_bytes

    def offload_to_cpu(self, module: nn.Module) -> int:
        """Offload module parameters to CPU and return bytes saved."""
        bytes_saved = 0

        # Log memory before offloading
        before_allocated = torch.cuda.memory_allocated()
        before_reserved = torch.cuda.memory_reserved()

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

                # Calculate bytes to be saved before copying
                param_size = param.data.numel() * param.data.element_size()
                # Copy data to CPU
                cpu_tensor.copy_(param.data)
                # Store reference to original data to explicitly delete it
                original_data = param.data
                # Update parameter data
                param.data = cpu_tensor
                # Explicitly delete the original tensor to free GPU memory
                del original_data
                # Update bytes saved
                bytes_saved += param_size
                # Update state
                self.param_states[name]["offloaded"] = True

        if bytes_saved > 0:
            # Force Python garbage collection to clean up any dangling references
            import gc

            gc.collect()
            # Now clear CUDA cache
            torch.cuda.empty_cache()

        return bytes_saved

    def load_to_gpu(self, module: nn.Module) -> int:
        """Load module parameters back to GPU and return bytes loaded."""
        bytes_loaded = 0
        successful_load = True

        # First try to load parameters
        for name, param in module.named_parameters():
            if param.device.type == "cpu":
                try:
                    # Move data back to original device
                    param.data = param.data.to(self.device, non_blocking=True)
                    # Calculate bytes loaded
                    bytes_loaded += param.data.numel() * param.data.element_size()
                    # Update state
                    self.param_states[name]["offloaded"] = False
                except RuntimeError as e:
                    # If we encounter a memory error, report it but don't crash
                    logger.error(f"GPU memory error when loading parameters: {e}")
                    # Ensure we don't lose the CPU copy if GPU loading fails
                    successful_load = False
                    torch.cuda.empty_cache()
                    return 0

        # If we've loaded parameters non-blocking, we need to synchronize to ensure they're ready
        if bytes_loaded > 0:
            # Verify all parameters are actually on GPU
            torch.cuda.synchronize()
            for name, param in module.named_parameters():
                if param.device != self.device:
                    logger.warning(
                        f"Parameter {name} failed to move to {self.device}, still on {param.device}"
                    )
                    successful_load = False

        if not successful_load:
            return 0

        return bytes_loaded


class ParameterOffloadManager:
    """Manager for parameter offloading between CPU and GPU memory."""

    def __init__(
        self,
        enable_offload: bool = False,
        enable_prefetch: bool = True,
        cpu_offload_ratio: float = 0.7,
        prefetch_window: int = 2,
        strict_device_match: bool = True,  # Whether to ensure strict device matching
    ):
        self.enable_offload = enable_offload
        self.enable_prefetch = enable_prefetch and enable_offload
        self.cpu_offload_ratio = cpu_offload_ratio
        self.prefetch_window = prefetch_window
        self.strict_device_match = strict_device_match

        # Store module information
        self.modules: Dict[str, nn.Module] = {}
        self.module_states: Dict[str, ModuleState] = {}
        self.usage_info: Dict[str, LayerUsageInfo] = defaultdict(LayerUsageInfo)
        self.execution_order: List[str] = []
        self.current_layer_idx: int = -1

        # Track offloaded bytes
        self.total_offloaded_bytes: int = 0
        self.total_parameter_bytes: int = 0

        # Lock for synchronization
        self.offload_lock = threading.Lock()

        # Prefetching queue and thread
        self.prefetch_queue = queue.Queue()
        self.prefetch_thread = None
        self.stop_prefetch = threading.Event()

        logger.info(
            f"Parameter offloading initialized (enabled={enable_offload}, prefetch={enable_prefetch}, strict_match={strict_device_match})"
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
                layer_num = 0  # Default value if we can't find a layer number
                parts = name.split(".")
                for i, part in enumerate(parts):
                    if (
                        part == layer_prefix
                        and i + 1 < len(parts)
                        and parts[i + 1].isdigit()
                    ):
                        layer_num = int(parts[i + 1])
                        break

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
            f"Offloaded {self.total_offloaded_bytes/1024/1024/1024:.2f}GB of parameters to CPU "
            f"({self.total_offloaded_bytes/self.total_parameter_bytes*100:.1f}% of total)"
        )
        torch.cuda.empty_cache()

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
            with self.offload_lock:
                module = self.modules[module_name]
                module_state = self.module_states[module_name]

                # Get size of module we want to load
                module_size = module_state.get_size_bytes()

                # Check if we need to free up GPU memory first
                free_memory = (
                    torch.cuda.get_device_properties(0).total_memory
                    - torch.cuda.memory_allocated()
                )

                # Add a buffer to account for fragmentation and other overhead
                required_memory = int(module_size * 1.2)  # 20% buffer

                # If we don't have enough free memory, offload other modules
                if free_memory < required_memory:
                    logger.info(
                        f"Need to free {(required_memory-free_memory)/1024/1024:.2f}MB for module {module_name}"
                    )
                    self._offload_least_used_modules(required_memory - free_memory)

                # Try to load the module to GPU
                try:
                    bytes_loaded = module_state.load_to_gpu(module)

                    if bytes_loaded > 0:
                        self.usage_info[module_name].on_gpu = True
                        logger.debug(
                            f"Loaded module {module_name} to GPU ({bytes_loaded/1024/1024:.2f}MB)"
                        )
                    else:
                        if self.strict_device_match:
                            # If strict matching is required and loading failed, we need to take corrective action
                            logger.warning(
                                f"Failed to load module {module_name} to GPU - still on CPU"
                            )
                            # Try again with more aggressive memory clearing
                            torch.cuda.empty_cache()
                            # Try to free even more memory
                            additional_memory = (
                                required_memory * 0.5
                            )  # Try to free 50% more
                            self._offload_least_used_modules(additional_memory)
                            # Try loading one more time
                            bytes_loaded = module_state.load_to_gpu(module)
                            if bytes_loaded > 0:
                                self.usage_info[module_name].on_gpu = True
                                logger.debug(
                                    f"Loaded module {module_name} to GPU on second attempt ({bytes_loaded/1024/1024:.2f}MB)"
                                )
                            else:
                                logger.warning(
                                    f"Still failed to load module {module_name} to GPU after second attempt"
                                )
                        else:
                            # If strict matching is not required, just log the warning
                            logger.warning(
                                f"Failed to load module {module_name} to GPU - still on CPU"
                            )
                except Exception as e:
                    logger.error(f"Error loading module {module_name} to GPU: {e}")

                # When strict device matching is required, ensure all tensors in module are on the correct device
                if self.strict_device_match:
                    # Verify all module parameters are on the correct device
                    for name, param in module.named_parameters():
                        expected_device = module_state.device
                        if param.device != expected_device:
                            logger.warning(
                                f"Parameter {name} is on wrong device: {param.device} (expected {expected_device})"
                            )
                            if hasattr(torch.cuda, "OutOfMemoryError"):
                                # This might fail with OOM, handle gracefully
                                try:
                                    # Force sync copy to ensure parameter is moved
                                    param.data = param.data.to(
                                        expected_device, non_blocking=False
                                    )
                                except Exception as e:
                                    logger.error(
                                        f"Couldn't move parameter {name} to {expected_device}: {e}"
                                    )

        # Request prefetch of next modules if prefetching is enabled
        if self.enable_prefetch and self.current_layer_idx >= 0:
            self._queue_prefetch()

    def _offload_least_used_modules(self, required_bytes: int) -> int:
        """Offload least recently used modules to free up GPU memory."""
        if not self.modules:
            return 0

        # Sort modules by last accessed time (oldest first) and whether they're on GPU
        candidates = [
            (name, info)
            for name, info in self.usage_info.items()
            if name in self.modules
            and info.on_gpu
            and name != self.execution_order[self.current_layer_idx]
        ]

        if not candidates:
            logger.warning("No candidate modules to offload")
            return 0

        # Sort by priority (low priority first) and last access time (oldest first)
        candidates.sort(key=lambda x: (x[1].priority_score, x[1].last_accessed))

        # Offload modules until we've freed enough memory
        bytes_freed = 0
        for name, _ in candidates:
            if bytes_freed >= required_bytes:
                break

            module = self.modules[name]
            module_state = self.module_states[name]

            logger.info(f"Offloading module {name} to make space")
            bytes_saved = module_state.offload_to_cpu(module)
            bytes_freed += bytes_saved
            self.usage_info[name].on_gpu = False
            self.total_offloaded_bytes += bytes_saved

        logger.info(
            f"Freed {bytes_freed/1024/1024:.2f}MB GPU memory by offloading modules"
        )
        return bytes_freed

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

                # Load module to GPU with memory management
                module = self.modules[module_name]
                module_state = self.module_states[module_name]

                # Get module size and check available memory
                module_size = module_state.get_size_bytes()
                free_memory = (
                    torch.cuda.get_device_properties(0).total_memory
                    - torch.cuda.memory_allocated()
                )
                required_memory = int(module_size * 1.2)  # Add 20% buffer

                # If we don't have enough memory, try to offload other modules
                if free_memory < required_memory:
                    logger.debug(
                        f"Prefetch: Need to free memory for module {module_name}"
                    )
                    try:
                        # Lock to prevent concurrent offloading from main thread
                        bytes_freed = self._offload_least_used_modules(
                            required_memory - free_memory
                        )
                        if bytes_freed < (required_memory - free_memory):
                            logger.warning(
                                f"Prefetch: Could not free enough memory for {module_name}"
                            )
                            self.usage_info[module_name].is_prefetching = False
                            self.prefetch_queue.task_done()
                            continue
                    except Exception as e:
                        logger.error(f"Error during prefetch memory management: {e}")
                        self.usage_info[module_name].is_prefetching = False
                        self.prefetch_queue.task_done()
                        continue

                # Try to load to GPU now that we've made space
                bytes_loaded = module_state.load_to_gpu(module)
                if bytes_loaded > 0:
                    # Update state
                    self.usage_info[module_name].on_gpu = True
                    logger.debug(
                        f"Prefetched module {module_name} to GPU ({bytes_loaded/1024/1024:.2f}MB)"
                    )
                else:
                    logger.warning(
                        f"Prefetch: Failed to load {module_name} to GPU despite freeing memory"
                    )

                # Reset prefetch flag
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
    strict_device_match: bool = True,  # Add new parameter
) -> ParameterOffloadManager:
    """Initialize the global parameter offload manager."""
    global parameter_offload_manager
    parameter_offload_manager = ParameterOffloadManager(
        enable_offload=enable_offload,
        enable_prefetch=enable_prefetch,
        cpu_offload_ratio=cpu_offload_ratio,
        prefetch_window=prefetch_window,
        strict_device_match=strict_device_match,  # Pass the parameter
    )
    return parameter_offload_manager


def get_parameter_offload_manager() -> Optional[ParameterOffloadManager]:
    """Get the global parameter offload manager."""
    return parameter_offload_manager
