from typing import Any, Dict, List, Optional
import torch
from scratchpad.nn.layers.linear import LinearMethodBase
from .base_config import QuantizationConfig
from triteia.python.nn.linear import sparse_low_precision_linear
from scratchpad.model_executor.parameter import (PackedvLLMParameter, ChannelQuantScaleParameter, BasevLLMParameter)

class TriteiaConfig(QuantizationConfig):
    def get_name(self) -> str:
        return "triteia_quant"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TriteiaConfig":
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        return TritelaLinearMethod(self)

    def get_scaled_act_names(self):
        return []
        

class TritelaLinearMethod(LinearMethodBase):
    
    def __init__(self, config: TriteiaConfig):
        self.config = config
    
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight_loader = extra_weight_attrs.get("weight_loader")
        output_size_per_partition = sum(output_partition_sizes)
        tile_size = 16
        pack_fator = 8
        self.layer = sparse_low_precision_linear(input_size_per_partition, output_size_per_partition)
        self.layer.qweight = PackedvLLMParameter(
            data=self.layer.qweight.data,
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=pack_fator,
            marlin_tile_size=tile_size,
            weight_loader=weight_loader
        )

        self.layer.meta = PackedvLLMParameter(
            data=self.layer.meta.data,
            input_dim=1,
            output_dim=0,
            packed_dim=1,
            packed_factor=1,
            marlin_tile_size=2,
            weight_loader=weight_loader
        )
        
        self.layer.scales = ChannelQuantScaleParameter(
            data=self.layer.scales.data,
            output_dim=1,
            weight_loader=weight_loader
        )
        
        self.layer.workspace = BasevLLMParameter(
            data=self.layer.workspace.data,
            weight_loader=weight_loader
        )
        
        layer.register_parameter("meta", self.layer.meta)
        layer.register_parameter("qweight", self.layer.qweight)
        layer.register_parameter("scales", self.layer.scales)

        
    def apply(self, layer, x, bias):
        return self.layer(x)        