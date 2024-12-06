"""deltazip model loader"""
from torch import nn
from .model_loader import (
    BaseModelLoader,
    LoadConfig,
    ModelConfig,
)
from scratchpad.utils import snapshot_download
import safetensors as st


class DeltazipModelLoader(BaseModelLoader):
    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        self.load_config = load_config
        self.pointer = None

    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        print(f"Downloading model {model_config.path}...")
        return snapshot_download(model_config.path)

    def load_model(self, *, model_config, device_config, cache_config):
        pass
