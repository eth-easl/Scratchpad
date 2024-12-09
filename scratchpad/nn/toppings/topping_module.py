import re
import os
import json
import torch
from torch import nn
import safetensors as st
from scratchpad.model_executor.model_loader import DefaultModelLoader
from scratchpad.model_executor.deltazip_loader import DeltazipModelLoader


class ToppingLayer(nn.Module):
    def __init__(self, config, base_hf_config):
        super().__init__()
        self.config = config
        self.base_hf_config = base_hf_config
        self.weights = {}
        self.weight_gpu = {}

    def load_to_gpu(self):
        for name, weight in self.weights.items():
            self.weight_gpu[name] = weight.to(torch.float16).to("cuda")

    def offload_from_gpu(self):
        for name, weight in self.weights.items():
            self.weight_gpu[name] = None


class ToppingAdapter(nn.Module):
    def __init__(self, uid, config, base_hf_config, load_config):
        super().__init__()
        self.uid = uid
        self.config = config
        self.base_hf_config = base_hf_config
        self.load_config = load_config

        self.layers = nn.ModuleList(
            [
                ToppingLayer(config, base_hf_config)
                for i in range(base_hf_config.num_hidden_layers)
            ]
        )
        self.weights = {}
        self.weights_gpu = {}

    def get_stacked_multiply(self, module_name):
        stacked_rank = {
            "qkv_proj": 3,
            "kv_proj": 2,
            "gate_up_proj": 2,
        }
        return stacked_rank[module_name] if module_name in stacked_rank else 1

    def load_to_gpu(self):
        for name, weight in self.weights.items():
            self.weights_gpu[name] = weight.to(torch.float16).to("cuda")
        for layer in self.layers:
            layer.load_to_gpu()

    def offload_from_gpu(self):
        for name, weight in self.weights.items():
            self.weights_gpu[name] = None
        for layer in self.layers:
            layer.offload_from_gpu()


class DeltaAdapter(ToppingAdapter):
    
    def __init__(self, uid, config, base_hf_config, load_config):
        super().__init__(uid, config, base_hf_config, load_config)
        self.pack_factor = None
        self.sparse_factor = None

    def get_stacked_multiply_delta(self, module_name):
        stacked_rank = {
            "kv_proj": 2,
            "gate_up_proj": 2,
        }
        return stacked_rank[module_name] if module_name in stacked_rank else 1
    
    def get_pack_factor(self):
        if self.pack_factor is None:
            raise ValueError("pack factor not initialized")
        return self.pack_factor

    def get_sparse_factor(self):
        if self.sparse_factor is None:
            raise ValueError("sparse factor not initialized")
        return self.sparse_factor

    def initialize_weights(self):
        print(f"Initializing weights...")
        loader = DeltazipModelLoader(self.load_config)
        local_path = loader.download_model(self.config)
        with open(os.path.join(local_path, "delta_config.json"), "r") as f:
            delta_config = json.load(f)
            self.pack_factor = 32 // delta_config["compress_config"]["bits"]
            self.sparse_factor = int(1 / delta_config["compress_config"]["sparsity"])
        weight_path = os.path.join(local_path, "deltazip-compressed.safetensors")
        with st.safe_open(weight_path, framework="torch", device="cpu") as f:
            keys = f.keys()
            for key in keys:
                match = re.search(r"layers\.(\d+)\.", key)
                if match is not None:
                    layer_id = int(match.group(1))
                    remaining_key = key[len(f"model.layers.{layer_id}.") :]
                    # TODO(xiaozhe): we need to get the correct rank
                    # let's assume rank==0 always
                    my_rank = 0
                    weight_val = f.get_tensor(key)
                    weight_name = remaining_key.replace(f".{my_rank}", "")
                    self.layers[layer_id].weights[weight_name] = weight_val.cpu()


class LoRAAdapter(ToppingAdapter):
    def __init__(self, uid, config, base_hf_config, load_config):
        super().__init__(uid, config, base_hf_config, load_config)
        self.scaling = self.config.hf_config["lora_alpha"] / self.config.hf_config["r"]

    def initialize_weights(self):
        model_path = self.config.path
        loader = DefaultModelLoader(self.load_config)
        revision = getattr(self.config.hf_config, "revision", None)
        for name, loaded_weight in loader._get_weights_iterator(
            DefaultModelLoader.Source(
                model_path, revision=revision, fall_back_to_pt=True
            )
        ):
            match = re.search(r"layers\.(\d+)\.", name)
            if match is not None:
                layer_id = int(match.group(1))
                self.layers[layer_id].weights[name] = loaded_weight.cpu()
            else:
                self.weights[name] = loaded_weight.cpu()

        # stack kv_proj and gate_up_proj
        for i in range(self.base_hf_config.num_hidden_layers):
            layer = self.layers[i]
            weight_names = [name for name, _ in layer.weights.items()]
            for weight_name in weight_names:
                if "k_proj" in weight_name:
                    q_name = weight_name.replace("k_proj", "q_proj")
                    v_name = weight_name.replace("k_proj", "v_proj")
                    kv_name = weight_name.replace("k_proj", "kv_proj")
                    qkv_name = weight_name.replace("k_proj", "qkv_proj")
                    if "lora_A" in weight_name:
                        layer.weights[qkv_name] = torch.cat(
                            (
                                layer.weights[q_name],
                                layer.weights[weight_name],
                                layer.weights[v_name],
                            ),
                            0,
                        )
                        layer.weights.pop(q_name)
                        layer.weights.pop(weight_name)
                        layer.weights.pop(v_name)
                    else:
                        layer.weights[kv_name] = torch.cat(
                            (
                                layer.weights[weight_name],
                                layer.weights[v_name],
                            ),
                            0,
                        )
                        layer.weights.pop(weight_name)
                        layer.weights.pop(v_name)
                elif "gate_proj" in weight_name:
                    up_name = weight_name.replace("gate_proj", "up_proj")
                    gate_up_name = weight_name.replace("gate_proj", "gate_up_proj")
                    layer.weights[gate_up_name] = torch.cat(
                        (layer.weights[weight_name], layer.weights[up_name]), 0
                    )
                    layer.weights.pop(weight_name)
                    layer.weights.pop(up_name)
