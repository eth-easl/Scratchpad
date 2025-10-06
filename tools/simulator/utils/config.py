from transformers import AutoConfig

def get_linear_layers_from_config(config: AutoConfig):
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    key_value_heads = config.num_key_value_heads
    attention_heads = config.num_attention_heads

    return get_linear_layers(
        hidden_size, intermediate_size, key_value_heads, attention_heads
    )


def get_linear_layers(
    hidden_size: int,
    intermediate_size: int,
    key_value_heads: int,
    attention_heads: int,
    tp_size: int = 1,
):
    if tp_size > 1:
        assert hidden_size % tp_size == 0
        assert intermediate_size % tp_size == 0
        assert key_value_heads % tp_size == 0

    return {
        "q_proj": [hidden_size, hidden_size // tp_size],
        "k_proj": [
            hidden_size,
            hidden_size * key_value_heads // attention_heads // tp_size,
        ],
        "v_proj": [
            hidden_size,
            hidden_size * key_value_heads // attention_heads // tp_size,
        ],
        "out_proj": [hidden_size // tp_size, hidden_size],
        "gate_proj": [hidden_size, intermediate_size // tp_size],
        "up_proj": [hidden_size, intermediate_size // tp_size],
        "down_proj": [intermediate_size // tp_size, hidden_size],
    }