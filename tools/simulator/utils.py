from transformers import AutoConfig


def flops_matmul(b, m, n, k, rank=None):
    # (b, m, r)
    if not rank:
        return 2 * b * m * n * k
    else:
        # (b, m)
        # (b, m, r) * (b, n, r)
        # (b, m, r) * (b, r, k)
        # there will be two matmul
        matmul_1 = 2 * b * m * rank
        matmul_2 = 2 * b * rank * k
        return matmul_1 + matmul_2


def memory_matmul(b, m, n, k, w_bit, a_bit, rank=None):
    # (b, m, n) * (b, n, k)
    if not rank:
        mem_load = w_bit / 8 * m * n
        activation_load = a_bit / 8 * b * m * k
        return mem_load, activation_load
    else:
        pass


def roofline_analyze(bandwidth, max_OPS, OPs, memory_access):
    # bandwidth is bytes/s
    # memory_access in byte
    # x axis is OPS/byte
    # y axis is OPS/s
    y_max = max_OPS
    memory_access_bytes = memory_access
    turning_point = y_max / bandwidth
    arithmetic_intensity = OPs / memory_access_bytes
    if arithmetic_intensity < turning_point:
        bound = "memory"
        performance = arithmetic_intensity * bandwidth
    else:
        bound = "compute"
        performance = y_max
    if performance == 0:
        pass
    return arithmetic_intensity, performance, bound


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
