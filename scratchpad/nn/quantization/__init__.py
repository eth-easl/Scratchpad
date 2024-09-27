from .base_config import QuantizationConfig, QuantizeMethodBase
from typing import Type

QUANTIZATION_METHODS = {}


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")
    return QUANTIZATION_METHODS[quantization]
