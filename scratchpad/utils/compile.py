import torch
import triton


def is_triton_3():
    return triton.__version__.startswith("3.")


def maybe_torch_compile(*args, **kwargs):
    """
    torch.compile does not work for triton 2.2.0, which is needed in xlm1's jax.
    Therefore, we disable it here.
    """

    def decorator(func):
        if is_triton_3():
            return torch.compile(*args, **kwargs)(func)
        return func

    return decorator
