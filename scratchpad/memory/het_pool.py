import torch
from scratchpad.utils import logger
from scratchpad.utils import (
    get_available_gpu_memory,
)

from .pool import BaseTokenToKVPool


class HeterogeneousMHATokenToKVPool(BaseTokenToKVPool):
    def __init__(
        self,
        size: int,
        cpu_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
    ):
        super().__init__(size, dtype)
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num

        # [size, head_num, head_dim] for each layer
        self.k_buffer = [
            torch.empty(
                (size + 1, head_num, head_dim), dtype=self.store_dtype, device="cuda"
            )
            for _ in range(layer_num)
        ]
        self.v_buffer = [
            torch.empty(
                (size + 1, head_num, head_dim), dtype=self.store_dtype, device="cuda"
            )
            for _ in range(layer_num)
        ]

        self.cpu_k_buffer = [
            torch.empty(
                (cpu_size + 1, head_num, head_dim), dtype=self.store_dtype, device="cpu"
            )
            for _ in range(layer_num)
        ]
        self.cpu_v_buffer = [
            torch.empty(
                (cpu_size + 1, head_num, head_dim), dtype=self.store_dtype, device="cpu"
            )
            for _ in range(layer_num)
        ]

    def get_key_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id].view(self.dtype)
        return self.k_buffer[layer_id]

    def get_value_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.v_buffer[layer_id].view(self.dtype)
        return self.v_buffer[layer_id]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
        if cache_v.dtype != self.dtype:
            cache_v = cache_v.to(self.dtype)
        if self.store_dtype != self.dtype:
            self.k_buffer[layer_id][loc] = cache_k.view(self.store_dtype)
            self.v_buffer[layer_id][loc] = cache_v.view(self.store_dtype)
        else:
            self.k_buffer[layer_id][loc] = cache_k
            self.v_buffer[layer_id][loc] = cache_v

    def shrink(self):
        pass

    def expand(self, increments, gpu_id):
        # expand the k_buffer and v_buffer by increments
        new_k_buffer = [
            torch.empty(
                (increments + 1, self.head_num, self.head_dim),
                dtype=self.store_dtype,
                device="cuda",
            )
            for _ in range(self.layer_num)
        ]
        new_v_buffer = [
            torch.empty(
                (increments + 1, self.head_num, self.head_dim),
                dtype=self.store_dtype,
                device="cuda",
            )
            for _ in range(self.layer_num)
        ]
        for i in range(self.layer_num):
            self.k_buffer[i] = torch.cat([self.k_buffer[i], new_k_buffer[i]], dim=0)
            self.v_buffer[i] = torch.cat([self.v_buffer[i], new_v_buffer[i]], dim=0)
        logger.info(
            f"Expand token kv pool to {self.size + increments}, avail mem={get_available_gpu_memory(gpu_id):.2f} GB"
        )
        self.size += increments
