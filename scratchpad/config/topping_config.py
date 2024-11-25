import os
import json
from enum import Enum, auto
from huggingface_hub import snapshot_download


class ToppingType(Enum):
    lora = auto()
    delta = auto()
    full = auto()


class ToppingConfig:
    def __init__(self, path: str, topping_type: ToppingType) -> None:
        self.path = path
        self.hf_config = self.get_topping_config()
        self.type: ToppingType = ToppingType[topping_type]

    def get_topping_config(self):
        if not os.path.isdir(self.path):
            weights_dir = snapshot_download(self.path, allow_patterns=["*.json"])
        else:
            weights_dir = self.path
        if self.type == ToppingType.lora:
            config_name = "adapter_config.json"
        elif self.type == ToppingType.delta:
            config_name = "delta_config.json"
        with open(os.path.join(weights_dir, config_name), "r") as f:
            return json.load(f)
