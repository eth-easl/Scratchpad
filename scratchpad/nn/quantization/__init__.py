from .base_config import QuantizationConfig, QuantizeMethodBase
from typing import Type

QUANTIZATION_METHODS = [
    "triteia"
]


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")
    from .triteia import TriteiaConfig
    quantization_mappings = {
        "triteia": TriteiaConfig
    }
    return quantization_mappings[quantization]

