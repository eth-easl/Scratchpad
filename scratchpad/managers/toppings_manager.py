import re
import torch
from scratchpad.server.args import ServerArgs
from scratchpad.utils import logger
from scratchpad.config.topping_config import ToppingType
from scratchpad.model_executor.forward_info import ForwardBatch
from scratchpad.utils.toppings.topping_utils import parse_topping_config
from scratchpad.memory.topping_pool import ToppingMemPool
from scratchpad.config.topping_config import ToppingConfig
from scratchpad.nn.toppings import LoRAAdapter, get_topping_layer
from scratchpad.utils import replace_submodule


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
        self.topping_memory_pool = ToppingMemPool(args)
        # TODO(xiaozhe): move below to topping memory pool
        # preallocate lora memory pool

        self.A_buffer = {}
        self.B_buffer = {}
        num_layer = self.base_hf_config.num_hidden_layers
        for module_A, module_B in self.target_weights:
            # init A tensor, column_major=True
            if hasattr(self.base_model, "get_hidden_dim"):
                hidden_dim_A, _ = self.base_model.get_hidden_dim(module_A)
            else:
                logger.warning(
                    "WARNING: get_hidden_dim() is not defined, "
                    "which is used to get the hidden dim for different lora modules"
                    "Use the default one, but please check if it is correct for your model."
                )
                hidden_dim_A, _ = get_hidden_dim(module_A, self.base_hf_config)
            c = self.loras[-1].get_stacked_multiply(module_A)
            if module_A not in self.A_buffer:
                self.A_buffer[module_A] = [
                    torch.empty(
                        (
                            self.max_toppings_per_batch,
                            self.max_lora_dim * c,
                            hidden_dim_A,
                        ),
                        dtype=self.dtype,
                        device="cuda",
                    )
                    for i in range(num_layer)
                ]
            # init B tensor, column_major=True
            if hasattr(self.base_model, "get_hidden_dim"):
                _, hidden_dim_B = self.base_model.get_hidden_dim(module_B)
            else:
                logger.warning(
                    "WARNING: get_hidden_dim() is not defined, "
                    "which is used to get the hidden dim for different lora modules"
                    "Use the default one, but please check if it is correct for your model."
                )
                _, hidden_dim_B = get_hidden_dim(module_B, self.base_hf_config)
            c = self.loras[-1].get_stacked_multiply(module_B)
            if module_B not in self.B_buffer:
                self.B_buffer[module_B] = [
                    torch.empty(
                        (
                            self.max_toppings_per_batch,
                            hidden_dim_B * c,
                            self.max_lora_dim,
                        ),
                        dtype=self.dtype,
                        device="cuda",
                    )
                    for i in range(num_layer)
                ]

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

        # load all weights to cpu
        self.loras = []
        self.lora_id = {}
        for name in self.available_toppings.keys():
            self.lora_id[name] = len(self.loras)
            self.loras.append(
                LoRAAdapter(
                    name, self.configs[name], self.base_hf_config, self.load_config
                )
            )
            self.loras[-1].initialize_weights()

        # misc lora configs
        self.max_lora_dim = max([x.hf_config["r"] for x in self.configs.values()])
        self.scaling = self.loras[0].scaling
        # FIXME remove the restrictions
        assert all(x.hf_config["r"] == self.max_lora_dim for x in self.configs.values())
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
        lora_module = get_topping_layer(module, None, self.max_lora_dim, self.scaling)
        replace_submodule(self.base_model, module_name, lora_module)
        return lora_module

    def prepare_topping_batch(self, forward_batch: ForwardBatch):
        print(f"Preparing topping batch for {forward_batch.topping_paths}")
        cur_uids = set(forward_batch.topping_paths)
        assert len(cur_uids) <= self.max_toppings_per_batch
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
        seg_lens = (
            forward_batch.extend_seq_lens
            if forward_batch.forward_mode.is_extend()
            else torch.ones(bs, device="cuda")
        )
        # FIXME: reuse the data rather than recompute
        seg_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device="cuda")
        seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)
        weight_indices = torch.empty((bs,), dtype=torch.int64, device="cuda")
        for i, topping_path in enumerate(forward_batch.topping_paths):
            weight_indices[i] = self.buffer_id[topping_path]

        for module_name, module in self.topping_modules:
            layer_id = get_layer_id(module_name)

            if "qkv_proj" not in module_name:
                weight_name = self.get_weight_name(module_name, 0)
                module.set_lora_info(
                    self.A_buffer[weight_name][layer_id],
                    self.B_buffer[weight_name][layer_id],
                    bs,
                    seg_indptr,
                    weight_indices,
                )
            else:
                module.set_lora_info(
                    self.A_buffer["qkv_proj"][layer_id],
                    self.B_buffer["q_proj"][layer_id],
                    self.B_buffer["kv_proj"][layer_id],
                    bs,
                    seg_indptr,
                    weight_indices,
                )

    def register_topping(
        self, topping_type: ToppingType, topping_path: str, topping_name: str
    ):
        self.available_toppings[topping_name] = (topping_type, topping_path)

    def allocate_memory(self):
        pass

    def load_topping(self, uid, buffer_id):
        print(f"Loading topping {uid} to buffer {buffer_id}")
        if uid not in self.available_toppings:
            raise ValueError(f"Topping {uid} not registered")
        print(f"type: {self.available_toppings[uid][0]}")
        # TODO(xiaozhe): check why we cannot use ToppingType.lora directly
        if self.available_toppings[uid][0] == "lora":
            self._load_lora(uid, buffer_id)
        else:
            raise NotImplementedError(
                f"Loading topping {uid} not implemented: Expected Type {ToppingType.lora}, got {self.available_toppings[uid][0]}"
            )

    def _load_lora(self, uid, buffer_id):
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
                        self.A_buffer[lora_weight_name][i][buffer_id].copy_(weights)
                else:
                    lora_weight_name = self.get_weight_name(name, 1)
                    if lora_weight_name:
                        self.B_buffer[lora_weight_name][i][buffer_id].copy_(weights)

    def _load_delta(self):
        pass

    def unload_topping(self):
        pass

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
