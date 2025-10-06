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