import torch
from typing import TYPE_CHECKING, List
from scratchpad.utils.logger import logger

if TYPE_CHECKING:
    from scratchpad.server.args import ServerArgs


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
    elif module_name == "lm_head":
        return config.vocab_size, config.hidden_size
    elif module_name == "logits_processor":
        return config.vocab_size, config.hidden_size
    else:
        print(f"get_hidden_dim for {module_name} is not implemented")
        raise NotImplementedError(
            f"get_hidden_dim for {module_name} is not implemented"
        )


class ToppingMemPool:
    def __init__(
        self,
        args: "ServerArgs",
        base_hf_config,
        loras: List,
        target_weights: List,
        max_toppings_per_batch: int,
        max_lora_dim: int,
        base_model,
        lora_dtype: torch.dtype,
        deltas: List,
        delta_target_weights: List,
        uncompressed_delta_target_weights: List,
    ):
        self.args = args
        self.base_model = base_model
        self.base_hf_config = base_hf_config
        self.loras = loras
        self.target_weights = target_weights
        self.max_toppings_per_batch = max_toppings_per_batch
        self.max_lora_dim = max_lora_dim
        self.lora_dtype = lora_dtype
        # for lora
        self.A_buffer = {}
        self.B_buffer = {}
        # for delta
        self.deltas = deltas
        self.delta_target_weights = delta_target_weights
        self.uncompressed_delta_target_weights = uncompressed_delta_target_weights
        self.qweight_buffer = {}
        self.meta_buffer = {}
        self.scales_buffer = {}
        self.weights_buffer = {}
        delta_dtypes = {
            "meta": torch.int16,
            "qweight": torch.int32,
            "scales": torch.float16,
        }

        # allocate lora
        num_layers = self.base_hf_config.num_hidden_layers
        for module_A, module_B in self.target_weights:
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
                            hidden_dim_A,
                            self.max_lora_dim * c,
                        ),
                        dtype=self.lora_dtype,
                        device="cuda",
                    )
                    for i in range(num_layers)
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
                            self.max_lora_dim,
                            hidden_dim_B * c,
                        ),
                        dtype=self.lora_dtype,
                        device="cuda",
                    )
                    for i in range(num_layers)
                ]

        # allocate delta
        pack_factor = self.deltas[-1].get_pack_factor()
        sparse_factor = self.deltas[-1].get_sparse_factor()
        for module in uncompressed_delta_target_weights:
            self.weights_buffer[module] = torch.zeros(
                self.max_toppings_per_batch,
                self.base_model.get_hidden_dim(module)[0],
                self.base_model.get_hidden_dim(module)[1],
                dtype=torch.bfloat16,
                device="cuda",
            )

        for module in delta_target_weights:
            dimensions = self.base_model.get_hidden_dim(module)
            stack_factor = self.deltas[-1].get_stacked_multiply_delta(module)
            stacked_dim = dimensions[1] * stack_factor

            self.qweight_buffer[module] = [
                torch.empty(
                    self.max_toppings_per_batch,
                    dimensions[0] // (pack_factor * sparse_factor * 2),
                    stacked_dim * 2,
                    dtype=delta_dtypes["qweight"],
                    device="cuda",
                )
                for i in range(num_layers)
            ]
            self.meta_buffer[module] = [
                torch.empty(
                    self.max_toppings_per_batch,
                    stacked_dim,
                    dimensions[0] // (pack_factor * sparse_factor),
                    dtype=delta_dtypes["meta"],
                    device="cuda",
                )
                for i in range(num_layers)
            ]
            self.scales_buffer[module] = [
                torch.empty(
                    self.max_toppings_per_batch,
                    1,
                    stacked_dim,
                    dtype=delta_dtypes["scales"],
                    device="cuda",
                )
                for i in range(num_layers)
            ]
