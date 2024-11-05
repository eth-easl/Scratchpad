import functools
import importlib
from typing import Dict, List, Optional, Tuple, Type

import torch.nn as nn
from scratchpad.utils import logger

_GENERATION_MODELS = {
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
    "LlamaNaiveQuantisedMoEForCausalLM": (
        "llama_naive_moe",
        "LlamaNaiveQuantisedMoEForCausalLM",
    ),
    "LlamaQuantisedMoEForCausalLM": ("llama_quant_moe", "LlamaQuantisedMoEForCausalLM"),
    "LlamaMoEForCausalLM": ("llama_moe", "LlamaMoEForCausalLM"),
}

_EMBEDDING_MODELS = {
    "MistralModel": ("llama_embedding", "LlamaEmbeddingModel"),
}

_MULTIMODAL_MODELS = {
    "MllamaForConditionalGeneration": ("mllama", "MllamaForConditionalGeneration"),
}
_CONDITIONAL_GENERATION_MODELS = {}

_MODELS = {
    **_GENERATION_MODELS,
    **_EMBEDDING_MODELS,
    **_MULTIMODAL_MODELS,
    **_CONDITIONAL_GENERATION_MODELS,
}
# Architecture -> type.
# out of tree models
_OOT_MODELS: Dict[str, Type[nn.Module]] = {}


class ModelRegistry:
    @staticmethod
    @functools.lru_cache(maxsize=128)
    def _get_model(model_arch: str):
        module_name, model_cls_name = _MODELS[model_arch]
        module = importlib.import_module(
            f"scratchpad.model_executor.models.{module_name}"
        )
        return getattr(module, model_cls_name, None)

    @staticmethod
    def _try_load_model_cls(model_arch: str) -> Optional[Type[nn.Module]]:
        if model_arch in _OOT_MODELS:
            return _OOT_MODELS[model_arch]
        if model_arch not in _MODELS:
            return None
        return ModelRegistry._get_model(model_arch)

    @staticmethod
    def resolve_model_cls(architectures: List[str]) -> Tuple[Type[nn.Module], str]:
        for arch in architectures:
            model_cls = ModelRegistry._try_load_model_cls(arch)
            if model_cls is not None:
                return (model_cls, arch)

        raise ValueError(
            f"Model architectures {architectures} are not supported for now. "
            f"Supported architectures: {ModelRegistry.get_supported_archs()}"
        )

    @staticmethod
    def get_supported_archs() -> List[str]:
        return list(_MODELS.keys()) + list(_OOT_MODELS.keys())

    @staticmethod
    def register_model(model_arch: str, model_cls: Type[nn.Module]):
        if model_arch in _MODELS:
            logger.warning(
                "Model architecture %s is already registered, and will be "
                "overwritten by the new model class %s.",
                model_arch,
                model_cls.__name__,
            )
        global _OOT_MODELS
        _OOT_MODELS[model_arch] = model_cls

    @staticmethod
    def is_embedding_model(model_arch: str) -> bool:
        return model_arch in _EMBEDDING_MODELS

    @staticmethod
    def is_multimodal_model(model_arch: str) -> bool:

        # TODO: find a way to avoid initializing CUDA prematurely to
        # use `supports_multimodal` to determine if a model is multimodal
        # model_cls = ModelRegistry._try_load_model_cls(model_arch)
        # from vllm.model_executor.models.interfaces import supports_multimodal
        return model_arch in _MULTIMODAL_MODELS


__all__ = [
    "ModelRegistry",
]
