import os
import glob
import torch
from torch import nn
import dataclasses
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Generator, cast, Iterable, List, Dict, Type, Any
from contextlib import contextmanager
from scratchpad.config import (
    LoadConfig,
    CacheConfig,
    ModelConfig,
    DeviceConfig,
    LoadFormat,
)
from scratchpad.utils import (
    download_weights_from_hf,
    is_pin_memory_available,
    download_safetensors_index_file_from_hf,
    current_platform,
)
from transformers import PretrainedConfig
from scratchpad.nn.quantization import QuantizationConfig
from scratchpad.config.modality_config import MultiModalConfig
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME
from .utils import (
    set_default_torch_dtype,
    pt_weights_iterator,
    filter_duplicate_safetensors_files,
    filter_files_not_needed_for_inference,
    safetensors_weights_iterator,
    get_model_architecture,
    get_quant_config,
)


@contextmanager
def device_loading_context(module: torch.nn.Module, target_device: torch.device):
    if target_device.type == "cpu":
        # If target is CPU, no need to move anything
        yield module
        return

    original_device_states: Dict[str, torch.device] = {}

    # Store original device states and move parameters to GPU if they're on CPU
    for name, p in module.named_parameters():
        if p.device.type == "cpu":
            original_device_states[name] = p.device
            p.data = p.data.to(target_device)
        # Parameters already on target device are not touched

    try:
        yield module

    finally:
        # Restore parameters to their original devices, ignoring new parameters
        pin_memory = is_pin_memory_available()
        for name, p in module.named_parameters():
            if name in original_device_states:
                original_device: torch.device = original_device_states[name]
                if original_device.type == "cpu":
                    # `torch.empty_like` does not support `pin_memory` argument
                    cpu_data = torch.empty_strided(
                        size=p.data.size(),
                        stride=p.data.stride(),
                        dtype=p.data.dtype,
                        layout=p.data.layout,
                        device="cpu",
                        pin_memory=pin_memory,
                    )
                    cpu_data.copy_(p.data)
                    p.data = cpu_data
                else:
                    p.data = p.data.to(original_device)
        # New parameters or parameters already on target device are untouched


class BaseModelLoader(ABC):
    """Base class for model loaders."""

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    @abstractmethod
    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    @abstractmethod
    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
    ) -> nn.Module:
        """Load a model with the given configurations."""
        raise NotImplementedError


def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Default weight loader."""
    try:
        if param.numel() == 1 and loaded_weight.numel() == 1:
            # Sometimes scalar values aren't considered tensors with shapes
            # so if both param and loaded_weight are a scalar,
            # "broadcast" instead of copy
            param.data.fill_(loaded_weight.item())
        else:
            assert param.size() == loaded_weight.size(), (
                f"Attempted to load weight ({loaded_weight.size()}) "
                f"into parameter ({param.size()})"
            )

            param.data.copy_(loaded_weight)
    except Exception:
        # NOTE: This exception is added for the purpose of setting breakpoint to
        # debug weight loading issues.
        raise


def _get_model_initialization_kwargs(
    model_class: Type[nn.Module],
    # lora_config: Optional[LoRAConfig],
    multimodal_config: Optional[MultiModalConfig],
) -> Dict[str, Any]:
    """Get extra kwargs for model initialization."""
    extra_kwargs: Dict[str, Any] = {}

    return extra_kwargs


def _get_quantization_config(
    model_config: ModelConfig, load_config: LoadConfig
) -> Optional[QuantizationConfig]:
    """Get the quantization config."""
    if model_config.quantization is not None:
        quant_config = get_quant_config(model_config, load_config)
        capability_tuple = current_platform.get_device_capability()

        if capability_tuple is not None:
            capability = capability_tuple.to_int()
            if capability < quant_config.get_min_capability():
                raise ValueError(
                    f"The quantization method {model_config.quantization} "
                    "is not supported for the current GPU. "
                    f"Minimum capability: {quant_config.get_min_capability()}. "
                    f"Current capability: {capability}."
                )
        supported_dtypes = quant_config.get_supported_act_dtypes()
        if model_config.dtype not in supported_dtypes:
            raise ValueError(
                f"{model_config.dtype} is not supported for quantization "
                f"method {model_config.quantization}. Supported dtypes: "
                f"{supported_dtypes}"
            )
        return quant_config
    return None


def build_model(
    model_class: Type[nn.Module],
    hf_config: PretrainedConfig,
    cache_config: Optional[CacheConfig],
    quant_config: Optional[QuantizationConfig],
    *,
    multimodal_config: Optional[MultiModalConfig],
) -> nn.Module:
    extra_kwargs = _get_model_initialization_kwargs(model_class, multimodal_config)

    return model_class(
        config=hf_config,
        cache_config=cache_config,
        quant_config=quant_config,
        **extra_kwargs,
    )


def _initialize_model(
    model_config: ModelConfig,
    load_config: LoadConfig,
    cache_config: CacheConfig,
) -> nn.Module:
    """Initialize a model with the given configurations."""
    model_class, _ = get_model_architecture(model_config)

    return build_model(
        model_class,
        model_config.hf_config,
        cache_config=cache_config,
        quant_config=_get_quantization_config(model_config, load_config),
        multimodal_config=model_config.multimodal_config,
    )


class DefaultModelLoader(BaseModelLoader):
    """Model loader that can load different file types from disk."""

    @dataclasses.dataclass
    class Source:
        """A source for weights."""

        model_or_path: str
        """The model ID or path."""

        revision: Optional[str]
        """The optional model revision."""

        prefix: str = ""
        """A prefix to prepend to all weights."""

        fall_back_to_pt: bool = True
        """Whether .pt weights can be used."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(
                f"Model loader extra config is not supported for "
                f"load format {load_config.load_format}"
            )

    def _prepare_weights(
        self, model_name_or_path: str, revision: Optional[str], fall_back_to_pt: bool
    ) -> Tuple[str, List[str], bool]:
        """Prepare weights for the model.

        If the model is not local, it will be downloaded."""
        is_local = os.path.isdir(model_name_or_path)
        load_format = self.load_config.load_format
        use_safetensors = False
        index_file = SAFE_WEIGHTS_INDEX_NAME
        # Some quantized models use .pt files for storing the weights.
        if load_format == LoadFormat.AUTO:
            allow_patterns = ["*.safetensors", "*.bin"]
        elif load_format == LoadFormat.SAFETENSORS:
            use_safetensors = True
            allow_patterns = ["*.safetensors"]
        elif load_format == LoadFormat.MISTRAL:
            use_safetensors = True
            allow_patterns = ["consolidated*.safetensors"]
            index_file = "consolidated.safetensors.index.json"
        elif load_format == LoadFormat.PT:
            allow_patterns = ["*.pt"]
        elif load_format == LoadFormat.NPCACHE:
            allow_patterns = ["*.bin"]
        else:
            raise ValueError(f"Unknown load_format: {load_format}")

        if fall_back_to_pt:
            allow_patterns += ["*.pt"]

        if not is_local:
            hf_folder = download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                allow_patterns,
                revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )
        else:
            hf_folder = model_name_or_path

        hf_weights_files: List[str] = []
        for pattern in allow_patterns:
            hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
            if len(hf_weights_files) > 0:
                if pattern == "*.safetensors":
                    use_safetensors = True
                break

        if use_safetensors:
            # For models like Mistral-7B-Instruct-v0.3
            # there are both sharded safetensors files and a consolidated
            # safetensors file. Using both breaks.
            # Here, we download the `model.safetensors.index.json` and filter
            # any files not found in the index.
            if not is_local:
                download_safetensors_index_file_from_hf(
                    model_name_or_path,
                    index_file,
                    self.load_config.download_dir,
                    revision,
                )
            hf_weights_files = filter_duplicate_safetensors_files(
                hf_weights_files, hf_folder, index_file
            )
        else:
            hf_weights_files = filter_files_not_needed_for_inference(hf_weights_files)

        if len(hf_weights_files) == 0:
            raise RuntimeError(
                f"Cannot find any model weights with `{model_name_or_path}`"
            )

        return hf_folder, hf_weights_files, use_safetensors

    def _get_weights_iterator(
        self, source: "Source"
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
            source.model_or_path, source.revision, source.fall_back_to_pt
        )
        if use_safetensors:
            weights_iterator = safetensors_weights_iterator(hf_weights_files)
        else:
            weights_iterator = pt_weights_iterator(hf_weights_files)

        # Apply the prefix.
        return ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)

    def _get_all_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:

        primary_weights = DefaultModelLoader.Source(
            model_config.model,
            model_config.revision,
            prefix="",
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
        )
        yield from self._get_weights_iterator(primary_weights)

        secondary_weights = cast(
            Iterable[DefaultModelLoader.Source], getattr(model, "secondary_weights", ())
        )
        for source in secondary_weights:
            yield from self._get_weights_iterator(source)

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(
            model_config.model, model_config.revision, fall_back_to_pt=True
        )

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
    ) -> nn.Module:
        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = _initialize_model(
                    model_config,
                    self.load_config,
                    cache_config,
                )

            model.load_weights(self._get_all_weights(model_config, model))

            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    # When quant methods need to process weights after loading
                    # (for repacking, quantizing, etc), they expect parameters
                    # to be on the global target device. This scope is for the
                    # case where cpu offloading is used, where we will move the
                    # parameters onto device for processing and back off after.
                    with device_loading_context(module, target_device):
                        quant_method.process_weights_after_loading(module)
        return model.eval()


def get_model_loader(load_config: LoadConfig) -> BaseModelLoader:
    """Get a model loader based on the load format."""

    if isinstance(load_config.load_format, type):
        return load_config.load_format(load_config)

    return DefaultModelLoader(load_config)


def get_model(
    *, model_config: ModelConfig, load_config: LoadConfig, device_config: DeviceConfig
) -> nn.Module:
    loader = get_model_loader(load_config)
    return loader.load_model(model_config=model_config, device_config=device_config)
