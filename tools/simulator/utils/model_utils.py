import csv
from functools import lru_cache
import json
from pathlib import Path
from typing import Dict

import humanize
from huggingface_hub import model_info

MODEL_SIZE_CACHE_PATH = (
    Path(__file__).resolve().parent.parent / "examples" / "model_size.csv"
)
MODEL_SIZE_CACHE_HEADER = ("model_id", "parameter_count")
DEFAULT_MODEL_SIZE_GB = 140.0
BYTES_PER_GB = 1_000_000_000


def _load_model_size_cache() -> Dict[str, int]:
    """Load cached model parameter counts from disk."""
    cache: Dict[str, int] = {}
    if not MODEL_SIZE_CACHE_PATH.exists():
        return cache

    with MODEL_SIZE_CACHE_PATH.open("r", newline="") as cache_file:
        reader = csv.reader(cache_file)
        for row in reader:
            if not row:
                continue
            if row[0] == MODEL_SIZE_CACHE_HEADER[0]:
                continue

            model_id = row[0].strip()
            if not model_id:
                continue

            try:
                parameter_count = int(row[1])
            except (IndexError, ValueError):
                continue

            cache[model_id] = parameter_count

    return cache


def _write_model_size_cache(cache: Dict[str, int]) -> None:
    """Persist model parameter counts to disk."""
    MODEL_SIZE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_SIZE_CACHE_PATH.open("w", newline="") as cache_file:
        writer = csv.writer(cache_file)
        writer.writerow(MODEL_SIZE_CACHE_HEADER)
        for model_id, parameter_count in sorted(cache.items()):
            writer.writerow([model_id, parameter_count])


def _fetch_model_parameter_count(model_id: str) -> int:
    """Fetch model parameter count from Hugging Face."""
    try:
        info = model_info(model_id)
    except Exception:
        return -1

    safetensors = getattr(info, "safetensors", None)
    total_params = None
    if safetensors is not None:
        total_params = getattr(safetensors, "total", None)
        if total_params is None and isinstance(safetensors, dict):
            total_params = safetensors.get("total")

    if total_params is None:
        total_params = getattr(info, "model_parameters", None)

    if total_params is None:
        return -1

    try:
        return int(total_params)
    except (TypeError, ValueError):
        return -1


@lru_cache(maxsize=None)
def get_model_parameter_count(model_id: str) -> int:
    """
    Retrieve the parameter count for a model, using an on-disk cache.

    Args:
        model_id: Hugging Face model identifier.

    Returns:
        Parameter count if available, otherwise -1.
    """
    cache = _load_model_size_cache()
    if model_id in cache:
        return cache[model_id]

    parameter_count = _fetch_model_parameter_count(model_id)
    cache[model_id] = parameter_count
    _write_model_size_cache(cache)

    return parameter_count


def get_model_size_gb(
    model_id: str,
    *,
    weight_bits: int = 16,
    default_size_gb: float = DEFAULT_MODEL_SIZE_GB,
) -> float:
    """
    Convert a model's parameter count into an approximate size in gigabytes.

    Args:
        model_id: Hugging Face model identifier.
        weight_bits: Bit precision used to store model weights.
        default_size_gb: Fallback size when parameter count is unavailable.

    Returns:
        Estimated model size in gigabytes.
    """
    parameter_count = get_model_parameter_count(model_id)
    if parameter_count <= 0 or weight_bits <= 0:
        return default_size_gb

    bytes_per_parameter = weight_bits / 8
    size_gb = (parameter_count * bytes_per_parameter) / BYTES_PER_GB

    return size_gb


if __name__ == "__main__":
    with open(".local/trace_full.jsonl", "r") as f:
        models = [json.loads(line)["model"] for line in f]
    models = sorted(set(models))
    for model in models:
        size_gb = get_model_size_gb(model)
        human_size = humanize.naturalsize(size_gb * BYTES_PER_GB, binary=True)
        print(f"{model}: {size_gb:.2f} GB ({human_size})")
