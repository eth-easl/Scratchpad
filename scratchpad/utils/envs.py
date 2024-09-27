import os
import tempfile
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    LOCAL_RANK: int = 0
    SP_RINGBUFFER_WARNING_INTERVAL: float = 60
    SP_HOST_IP: str = ""
    SP_PORT: Optional[int] = None
    SP_VERBOSITY: int = 0
    SP_FLASH_INFER_WORKSPACE_SIZE: int = 384 * 1024 * 1024
    SP_ALLOW_LONG_MAX_MODEL_LEN: bool = False


def get_default_cache_root():
    return os.getenv(
        "XDG_CACHE_HOME",
        os.path.join(os.path.expanduser("~"), ".cache"),
    )


def get_default_config_root():
    return os.getenv(
        "XDG_CONFIG_HOME",
        os.path.join(os.path.expanduser("~"), ".config"),
    )


environment_variables: Dict[str, Callable[[], Any]] = {
    "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
    # Number of threads to use for nvcc
    # By default this is 1.
    # If set, `MAX_JOBS` will be reduced to avoid oversubscribing the CPU.
    "NVCC_THREADS": lambda: os.getenv("NVCC_THREADS", None),
    "LOCAL_RANK": lambda: os.getenv("LOCAL_RANK", 0),
    "SP_RINGBUFFER_WARNING_INTERVAL": lambda: os.getenv(
        "SP_RINGBUFFER_WARNING_INTERVAL", 60
    ),
    "SP_HOST_IP": lambda: os.getenv("SP_HOST_IP", ""),
    "SP_PORT": lambda: os.getenv("SP_PORT", None),
    "SP_VERBOSITY": lambda: os.getenv("SP_VERBOSITY", 0),
    "SP_FLASH_INFER_WORKSPACE_SIZE": lambda: os.getenv(
        "SP_FLASH_INFER_WORKSPACE_SIZE", 384 * 1024 * 1024
    ),
    "SP_ALLOW_LONG_MAX_MODEL_LEN": lambda: os.getenv(
        "SP_ALLOW_LONG_MAX_MODEL_LEN", False
    ),
    "SP_CACHE_ROOT": get_default_cache_root,
    "SP_CONFIG_ROOT": get_default_config_root,
    "SP_TEMP_DIR": lambda: tempfile.gettempdir(),
    "SP_LOG_DIR": lambda: os.getenv("SP_LOG_DIR", "/tmp"),
    "SP_LOG_FILE": lambda: os.getenv("SP_LOG_FILE", ""),
    "SP_LOG_LEVEL": lambda: os.getenv("SP_LOG_LEVEL", "INFO"),
    "SP_LOG_FORMAT": lambda: os.getenv(
        "SP_LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ),
}


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())
