import torch
from functools import lru_cache
from typing import Optional
from . import envs


def supports_custom_op() -> bool:
    return hasattr(torch.library, "custom_op")


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
