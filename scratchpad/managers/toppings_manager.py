import os
import re
import torch
from scratchpad.server.args import ServerArgs
from scratchpad.utils import logger
from scratchpad.config.topping_config import ToppingType
from scratchpad.model_executor.forward_info import ForwardBatch
from scratchpad.utils.toppings.topping_utils import parse_topping_config
from scratchpad.memory.topping_pool import ToppingMemPool
from scratchpad.config.topping_config import ToppingConfig
from scratchpad.nn.toppings import LoRAAdapter, DeltaAdapter, get_topping_layer
from scratchpad.utils import replace_submodule

num_replicated_lora = os.environ.get("SPB_NUM_REPLICATED_LORA", 1)
num_replicated_delta = os.environ.get("SPB_NUM_REPLICATED_DELTA", 1)


def get_layer_id(name):
    match = re.search(r"layers\.(\d+)\.", name)
    if match is None:
        return None
    return int(match.group(1))


def get_module_name(name):
    # Fallback solution of mapping from config module name to module name in model class.
    # Please check if it aligns with your base model.
    # Please implement the function in the model class if it is not.
    # You can reference this function in llama.py.
    params_mapping = {
        "q_proj": "qkv_proj",
        "k_proj": "qkv_proj",
        "v_proj": "qkv_proj",
        "gate_proj": "gate_up_proj",
        "up_proj": "gate_up_proj",
    }
    return params_mapping.get(name, name)


def get_hidden_dim(module_name, config):
    # Fallback solution of get_hidden_dim for different modules
    # Please check if it aligns with your base model.
    # Please implement the function in the model class if it is not.
    # You can reference this function in llama.py.
    if module_name in ["q_proj", "o_proj", "qkv_proj"]:
        return config.hidden_size, config.hidden_size
    elif module_name in ["kv_proj"]:
        return config.hidden_size, config.hidden_size // (
            config.num_attention_heads // config.num_key_value_heads
        )
    elif module_name == "gate_up_proj":
        return config.hidden_size, config.intermediate_size
    elif module_name == "down_proj":
        return config.intermediate_size, config.hidden_size
    else:
        raise NotImplementedError()


def get_stacked_name(name):
    # origin name -> (name for A, name for B)
    params_mapping = {
        "q_proj": ("qkv_proj", "q_proj"),
        "k_proj": ("qkv_proj", "kv_proj"),
        "v_proj": ("qkv_proj", "kv_proj"),
        "gate_proj": ("gate_up_proj", "gate_up_proj"),
        "up_proj": ("gate_up_proj", "gate_up_proj"),
    }
    return params_mapping.get(name, (name, name))


class ToppingsManager:
    def __init__(
        self,
        server_args: ServerArgs,
        base_model,
        base_hf_config,
        load_config,
    ):
        self.available_toppings = {}
        self.dtype = torch.bfloat16
        self.base_model = base_model
        self.base_hf_config = base_hf_config
        self.load_config = load_config
        self.toppings_id = {}
        self.max_toppings_per_batch = server_args.max_toppings_per_batch
        toppings = parse_topping_config(server_args.init_toppings)
        for topping in toppings:
            self.register_topping(topping[0], topping[1], topping[2])
        self.print_available_toppings()
        self.init_topping_batch()
        self.init_toppings()
        self.init_topping_mem_pool(server_args)
        logger.info("Topping manager ready.")

    def init_topping_mem_pool(self, args):
        self.topping_memory_pool = ToppingMemPool(
            args,
            base_hf_config=self.base_hf_config,
            loras=self.loras,
            target_weights=self.target_weights,
            max_toppings_per_batch=self.max_toppings_per_batch,
            base_model=self.base_model,
            max_lora_dim=self.max_lora_dim,
            lora_dtype=self.dtype,
            deltas=self.deltas,
            delta_target_weights=self.delta_target_weights,
        )

    def init_topping_batch(self):
        self.active_uids = set()  # set of active loras
        self.buffer_id = {}  # lora uid -> idx in memory pool

    def init_toppings(self):
        # get configs and target modules
        self.configs = {}
        self.origin_target_modules = set()
        for name, top in self.available_toppings.items():
            self.configs[name] = ToppingConfig(topping_type=top[0], path=top[1])

            self.origin_target_modules = set(self.origin_target_modules) | set(
                self.configs[name].hf_config["target_modules"]
            )

        if hasattr(self.base_model, "get_module_name"):
            self.target_modules = {
                self.base_model.get_module_name(module)
                for module in self.origin_target_modules
            }
        else:
            logger.warning(
                "WARNING: get_module_name() is not defined, "
                "which is used to map config module name to model implementation module name."
                "Use the default one, but please check if it is correct for your model."
            )
            self.target_modules = {
                get_module_name(module) for module in self.origin_target_modules
            }
        self.target_weights = set(
            [get_stacked_name(module) for module in self.origin_target_modules]
        )

        self.delta_target_weights = [
            "q_proj",
            "kv_proj",
            "gate_up_proj",
            "o_proj",
            "down_proj",
        ]

        # load all weights to cpu
        self.loras = []
        self.lora_id = {}
        self.deltas = []
        self.delta_id = {}
        for name in self.available_toppings.keys():
            t_type = self.available_toppings[name][0]
            logger.info(f"Loading {t_type} {name}")
            if t_type == "lora":
                self.lora_id[name] = len(self.loras)
                self.toppings_id[name] = len(self.loras)
                self.loras.append(
                    LoRAAdapter(
                        name, self.configs[name], self.base_hf_config, self.load_config
                    )
                )
                self.loras[-1].initialize_weights()
            elif t_type == "delta":
                self.delta_id[name] = len(self.deltas)
                self.toppings_id[name] = len(self.loras) + len(self.deltas)
                self.deltas.append(
                    DeltaAdapter(
                        name, self.configs[name], self.base_hf_config, self.load_config
                    )
                )
                self.deltas[-1].initialize_weights()

        # misc lora configs
        self.max_lora_dim = max(
            [x.hf_config["r"] for x in self.configs.values() if "r" in x.hf_config]
        )
        self.scaling = self.loras[0].scaling
        # FIXME remove the restrictions
        assert all(
            x.hf_config["r"] == self.max_lora_dim
            for x in self.configs.values()
            if "r" in x.hf_config
        )
        assert all(x.scaling == self.scaling for x in self.loras)

        # monkey patch to use the LoRA version
        self.topping_modules = []
        for module_name, module in self.get_target_modules():
            self.topping_modules.append(
                (module_name, self.set_lora_module(module_name, module))
            )

    def print_available_toppings(self):
        logger.info("Available toppings:")
        for topping in self.available_toppings:
            logger.info(f"({self.available_toppings[topping][0]}) {topping}")

    def set_lora_module(self, module_name, module):
        lora_module = get_topping_layer(module)
        replace_submodule(self.base_model, module_name, lora_module)
        return lora_module

    def prepare_topping_batch(self, forward_batch: ForwardBatch):
        cur_uids = set(forward_batch.topping_paths)
        assert (
            len(cur_uids) <= self.max_toppings_per_batch
        ), f"Got {len(cur_uids)} toppings, but max is {self.max_toppings_per_batch}"
        i = 0
        j = len(self.active_uids)
        evictable_uids = list(self.active_uids)
        for uid in cur_uids:
            if uid not in self.active_uids:
                if j < self.max_toppings_per_batch:
                    index = j
                    j += 1
                else:
                    while i < len(evictable_uids) and evictable_uids[i] in cur_uids:
                        i += 1
                    assert i < len(evictable_uids)
                    self.active_uids.remove(evictable_uids[i])
                    self.buffer_id.pop(evictable_uids[i])
                    index = i
                    i += 1
                self.load_topping(uid, index)
                self.active_uids.add(uid)
                self.buffer_id[uid] = index

        if cur_uids == set([None]):
            return

        # setup lora in forward modules
        bs = forward_batch.batch_size
        indices_len = forward_batch.input_ids.size(0)

        assert bs == len(
            forward_batch.topping_paths
        ), f"Expected batch size to match topping paths, got (bs={bs}) != ({len(forward_batch.topping_paths)})"

        if forward_batch.forward_mode == 1:  # prefill
            assert bs == 1, "Prefill mode only supports batch size 1"
            weight_indices = torch.full(
                (indices_len,),
                fill_value=self.toppings_id[forward_batch.topping_paths[0]],
                dtype=torch.int64,
                device="cuda",
            )
        else:
            # 1. Convert each request's topping_path to a topping_id
            weight_indices = torch.tensor(
                [self.toppings_id[tp] for tp in forward_batch.topping_paths],
                dtype=torch.int64,
                device=forward_batch.input_ids.device,
            )
        print(f"Weight indices: {weight_indices}")
        for module_name, module in self.topping_modules:
            layer_id = get_layer_id(module_name)
            if "qkv_proj" not in module_name:
                weight_name = self.get_weight_name(module_name, 0)
                module.set_topping_info(
                    bs,
                    weight_indices,
                    lora_buffer=(
                        self.A_buffer[weight_name][layer_id],
                        self.B_buffer[weight_name][layer_id],
                    ),
                    delta_buffer=(
                        self.qweight_buffer[weight_name][layer_id],
                        self.scales_buffer[weight_name][layer_id],
                        self.meta_buffer[weight_name][layer_id],
                    ),
                )
            else:
                module.set_topping_info(
                    bs,
                    weight_indices,
                    lora_buffer=(
                        self.A_buffer["qkv_proj"][layer_id],
                        self.B_buffer["q_proj"][layer_id],
                        self.B_buffer["kv_proj"][layer_id],
                    ),
                    delta_buffer_q=(
                        self.qweight_buffer["q_proj"][layer_id],
                        self.scales_buffer["q_proj"][layer_id],
                        self.meta_buffer["q_proj"][layer_id],
                    ),
                    delta_buffer_kv=(
                        self.qweight_buffer["kv_proj"][layer_id],
                        self.scales_buffer["kv_proj"][layer_id],
                        self.meta_buffer["kv_proj"][layer_id],
                    ),
                )

    def register_topping(
        self, topping_type: ToppingType, topping_path: str, topping_name: str
    ):
        self.available_toppings[topping_name] = (topping_type, topping_path)

    def allocate_memory(self, new_memory_size: int):
        pass

    def load_topping(self, uid, buffer_id):
        """
        This function loads topping from CPU -> GPU memory
        """
        if uid not in self.available_toppings:
            logger.error(f"Topping {uid} not registered")
            return
        if self.available_toppings[uid][0] == "lora":
            self._load_lora(uid, buffer_id)
        elif self.available_toppings[uid][0] == "delta":
            self._load_delta(uid, buffer_id)
        else:
            raise NotImplementedError(
                f"Loading topping {uid} not implemented: Expected Type {ToppingType.lora}, got {self.available_toppings[uid][0]}"
            )

    def _load_lora(self, uid, buffer_id):
        print(f"Loading LoRA {uid}")
        num_layer = self.base_hf_config.num_hidden_layers
        if uid is None:
            for i in range(num_layer):
                for k in self.A_buffer.keys():
                    self.A_buffer[k][i][buffer_id] *= 0
            return

        for i in range(num_layer):
            layer_weights = self.loras[self.lora_id[uid]].layers[i].weights
            for name, weights in layer_weights.items():
                if "lora_A" in name:
                    lora_weight_name = self.get_weight_name(name, 0)
                    if lora_weight_name:
                        self.A_buffer[lora_weight_name][i][buffer_id].copy_(weights.T)
                else:
                    lora_weight_name = self.get_weight_name(name, 1)
                    if lora_weight_name:
                        self.B_buffer[lora_weight_name][i][buffer_id].copy_(weights.T)

    def _load_delta(self, uid, buffer_id):
        print(uid, buffer_id)
        print(f"Loading Delta {uid}")
        num_layer = self.base_hf_config.num_hidden_layers

        if uid is None:
            for i in range(num_layer):
                for k in self.qweight_buffer.keys():
                    self.qweight_buffer[k][i][buffer_id] *= 0
            return

        for i in range(num_layer):
            layer_weights = self.deltas[self.delta_id[uid]].layers[i].weights
            for name, weights in layer_weights.items():
                if (
                    "qkv_proj" in name
                ):  # we need to extract, the q_proj and kv_proj slices
                    if "qweight" in name:
                        q_proj_name = "q_proj"
                        kv_proj_name = "kv_proj"
                        q_dim = self.qweight_buffer[q_proj_name][i][buffer_id].shape[1]
                        self.qweight_buffer[q_proj_name][i][buffer_id].copy_(
                            weights[:, :q_dim]
                        )
                        self.qweight_buffer[kv_proj_name][i][buffer_id].copy_(
                            weights[:, q_dim:]
                        )

                    elif "scales" in name:
                        q_proj_name = "q_proj"
                        kv_proj_name = "kv_proj"
                        q_dim = self.scales_buffer[q_proj_name][i][buffer_id].shape[1]
                        self.scales_buffer[q_proj_name][i][buffer_id].copy_(
                            weights[:, :q_dim]
                        )
                        self.scales_buffer[kv_proj_name][i][buffer_id].copy_(
                            weights[:, q_dim:]
                        )
                    else:
                        q_proj_name = "q_proj"
                        kv_proj_name = "kv_proj"
                        q_dim = self.meta_buffer[q_proj_name][i][buffer_id].shape[0]
                        self.meta_buffer[q_proj_name][i][buffer_id].copy_(
                            weights[:q_dim, :]
                        )
                        self.meta_buffer[kv_proj_name][i][buffer_id].copy_(
                            weights[q_dim:, :]
                        )
                else:  # the other layers can be used as such
                    if "qweight" in name:
                        weight_name = self.get_delta_weight_name(name)
                        if weight_name:
                            self.qweight_buffer[weight_name][i][buffer_id].copy_(
                                weights
                            )
                    elif "scales" in name:
                        weight_name = self.get_delta_weight_name(name)
                        if weight_name:
                            self.scales_buffer[weight_name][i][buffer_id].copy_(weights)
                    else:
                        weight_name = self.get_delta_weight_name(name)
                        if weight_name:
                            self.meta_buffer[weight_name][i][buffer_id].copy_(weights)

    def unload_topping(self):
        pass

    def get_delta_weight_name(self, name):
        for target_weight_name in self.delta_target_weights:
            if target_weight_name in name:
                return target_weight_name

    def get_weight_name(self, name, idx):
        for target_weight_name in self.target_weights:
            if target_weight_name[idx] in name:
                return target_weight_name[idx]

    def match_target_modules(self, module_name):
        for target_module in self.target_modules:
            if module_name.split(".")[-1] == target_module:
                return True
        return False

    def get_target_modules(self):
        modules = []
        for module_name, module in self.base_model.named_modules():
            if self.match_target_modules(module_name):
                modules.append((module_name, module))
        return modules

    @property
    def toppings(self):
        return list(self.available_toppings.keys())

    @property
    def A_buffer(self):
        return self.topping_memory_pool.A_buffer

    @property
    def B_buffer(self):
        return self.topping_memory_pool.B_buffer

    @property
    def qweight_buffer(self):
        return self.topping_memory_pool.qweight_buffer

    @property
    def meta_buffer(self):
        return self.topping_memory_pool.meta_buffer

    @property
    def scales_buffer(self):
        return self.topping_memory_pool.scales_buffer
