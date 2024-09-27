import os
import torch
from functools import lru_cache
from typing import Optional, Dict
import socket
import ipaddress
import sys
import traceback
from . import envs, logger
import random
import numpy as np
import psutil


def supports_custom_op() -> bool:
    return hasattr(torch.library, "custom_op")


def get_available_gpu_memory(gpu_id, distributed=False):
    """
    Get available memory for cuda:gpu_id device.
    When distributed is True, the available memory is the minimum available memory of all GPUs.
    """
    num_gpus = torch.cuda.device_count()
    assert gpu_id < num_gpus

    if torch.cuda.current_device() != gpu_id:
        print(
            f"WARNING: current device is not {gpu_id}, but {torch.cuda.current_device()}, ",
            "which may cause useless memory allocation for torch CUDA context.",
        )

    torch.cuda.empty_cache()
    free_gpu_memory, _ = torch.cuda.mem_get_info(gpu_id)

    if distributed:
        tensor = torch.tensor(free_gpu_memory, dtype=torch.float32).to(
            torch.device("cuda", gpu_id)
        )
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
        free_gpu_memory = tensor.item()

    return free_gpu_memory / (1 << 30)


@lru_cache(maxsize=8)
def _cuda_device_count_stateless(cuda_visible_devices: Optional[str] = None) -> int:
    # Note: cuda_visible_devices is not used, but we keep it as an argument for
    # LRU Cache purposes.

    # Code below is based on
    # https://github.com/pytorch/pytorch/blob/
    # c1cd946818442aca8c7f812b16d187ce1586c3bc/
    # torch/cuda/__init__.py#L831C1-L831C17
    import torch.cuda
    import torch.version

    if not torch.cuda._is_compiled():
        return 0
    raw_count = torch.cuda._device_count_nvml()
    r = torch._C._cuda_getDeviceCount() if raw_count < 0 else raw_count
    return r


def cuda_device_count_stateless() -> int:
    """Get number of CUDA devices, caching based on the value of
    CUDA_VISIBLE_DEVICES at the time of call.

    This should be used instead of torch.cuda.device_count()
    unless CUDA_VISIBLE_DEVICES has already been set to the desired
    value."""

    # This can be removed and simply replaced with torch.cuda.get_device_count
    # after https://github.com/pytorch/pytorch/pull/122815 is released.
    return _cuda_device_count_stateless(envs.CUDA_VISIBLE_DEVICES)


def update_environment_variables(envs: Dict[str, str]):
    for k, v in envs.items():
        if k in os.environ and os.environ[k] != v:
            logger.warning(
                "Overwriting environment variable %s " "from '%s' to '%s'",
                k,
                os.environ[k],
                v,
            )
        os.environ[k] = v


def find_nccl_library() -> str:
    if torch.version.cuda is not None:
        so_file = "libnccl.so.2"
    elif torch.version.hip is not None:
        so_file = "librccl.so.1"
    else:
        raise ValueError("NCCL only supports CUDA and ROCm backends.")
    logger.info("Found nccl from library %s", so_file)
    return so_file


def get_ip() -> str:
    host_ip = envs.SP_HOST_IP
    if host_ip:
        return host_ip

    # IP is not set, try to get it from the network interface

    # try ipv4
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        pass

    # try ipv6
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        # Google's public DNS server, see
        # https://developers.google.com/speed/public-dns/docs/using#addresses
        s.connect(("2001:4860:4860::8888", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        pass

    logger.warning(
        "Failed to get the IP address, using 0.0.0.0 by default."
        "The value can be set by the environment variable"
        " SP_HOST_IP.",
        stacklevel=2,
    )
    return "0.0.0.0"


def get_open_port() -> int:
    port = envs.SP_PORT
    if port is not None:
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", port))
                    return port
            except OSError:
                port += 1  # Increment port number if already in use
                logger.info("Port %d is already in use, trying port %d", port - 1, port)
    # try ipv4
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        # try ipv6
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]


def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def get_exception_traceback():
    etype, value, tb = sys.exc_info()
    err_str = "".join(traceback.format_exception(etype, value, tb))
    return err_str


def is_generation_model(model_architectures, is_embedding: bool = False):
    # We have two ways to determine whether a model is a generative model.
    # 1. Check the model architectue
    # 2. check the `is_embedding` server args

    if (
        "LlamaEmbeddingModel" in model_architectures
        or "MistralModel" in model_architectures
    ):
        return False
    else:
        return not is_embedding


def is_multimodal_model(model_architectures):
    if (
        "LlavaLlamaForCausalLM" in model_architectures
        or "LlavaQwenForCausalLM" in model_architectures
        or "LlavaMistralForCausalLM" in model_architectures
        or "LlavaVidForCausalLM" in model_architectures
    ):
        return True
    else:
        return False


@lru_cache
def print_warning_once(msg: str) -> None:
    # Set the stacklevel to 2 to print the caller's line info
    logger.warning(msg, stacklevel=2)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


def set_random_seed(seed: int) -> None:
    """Set the random seed for all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def kill_parent_process():
    """Kill the parent process and all children of the parent process."""
    current_process = psutil.Process()
    parent_process = current_process.parent()
    kill_child_process(parent_process.pid, skip_pid=current_process.pid)


def kill_child_process(pid, including_parent=True, skip_pid=None):
    """Kill the process and all its children process."""
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    children = parent.children(recursive=True)
    for child in children:
        if child.pid == skip_pid:
            continue
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    if including_parent:
        try:
            parent.kill()
        except psutil.NoSuchProcess:
            pass


@lru_cache(maxsize=None)
def is_pin_memory_available() -> bool:
    return True


def monkey_patch_vllm_all_gather(reverse: bool = False):
    """Monkey patch all-gather to remove in-place operations."""
    from torch.distributed import _functional_collectives as funcol
    from scratchpad.distributed.parallel_state import GroupCoordinator

    global vllm_all_gather_backup
    if vllm_all_gather_backup is None:
        vllm_all_gather_backup = GroupCoordinator.all_gather

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        assert (
            -input_.dim() <= dim < input_.dim()
        ), f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        input_size = input_.size()
        # Allocate output tensor.
        output_tensor = torch.empty(
            (world_size,) + input_size, dtype=input_.dtype, device=input_.device
        )

        output_tensor = funcol.all_gather_tensor(
            input_, gather_dim=0, group=self.device_group
        ).view((world_size,) + input_size)

        # Reshape
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(
            input_size[:dim] + (world_size * input_size[dim],) + input_size[dim + 1 :]
        )
        return output_tensor

    if reverse:
        setattr(GroupCoordinator, "all_gather", vllm_all_gather_backup)
    else:
        setattr(GroupCoordinator, "all_gather", all_gather)
