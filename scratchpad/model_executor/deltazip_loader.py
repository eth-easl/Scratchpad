"""deltazip model loader"""
from torch import nn
from .model_loader import (
    BaseModelLoader,
    LoadConfig,
    ModelConfig,
    DeviceConfig,
    CacheConfig,
)
from scratchpad.utils import snapshot_download


class DeltazipModelLoader(BaseModelLoader):
    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        print(f"Downloading model {model_config.path}...")
        return snapshot_download(model_config.path)

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
    ) -> nn.Module:
        """Load a model with the given configurations."""
        raise NotImplementedError
