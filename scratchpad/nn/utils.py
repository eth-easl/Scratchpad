import torch
from typing import Any, Dict, Optional
import torch
import triton
import triton.language as tl
from typing import Dict, Set


def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: Optional[Dict[str, Any]],
):
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor.
        weight_attrs: A dictionary of attributes to set on the weight tensor.
    """
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(weight, key), f"Overwriting existing tensor attribute: {key}"
        setattr(weight, key, value)


# ==== FlashInfer Utils ====


@triton.jit
def create_flashinfer_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_indptr,
    kv_start_idx,
    kv_indices_ptr,
    max_context_len: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)

    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start
    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    req_to_token_ptr += req_pool_index * max_context_len
    kv_indices_ptr += kv_indices_offset

    ld_offset = kv_start + tl.arange(0, BLOCK_SIZE)
    st_offset = tl.arange(0, BLOCK_SIZE)
    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = ld_offset < kv_end
        data = tl.load(req_to_token_ptr + ld_offset, mask=mask)
        tl.store(kv_indices_ptr + st_offset, data, mask=mask)
        ld_offset += BLOCK_SIZE
        st_offset += BLOCK_SIZE


class FlashinferUpdater:
    def __init__(
        self,
        forward_mode,
        model_runner,
        req_pool_indices,
        seq_lens,
        prefix_lens,
        decode_wrapper=None,
        use_ragged=False,
    ):
        self.forward_mode = forward_mode
        self.model_runner = model_runner
        self.req_pool_indices = req_pool_indices
        self.seq_lens = seq_lens
        self.prefix_lens = prefix_lens
        self.use_ragged = use_ragged

        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // model_runner.tp_size
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            model_runner.tp_size
        )
        self.head_dim = model_runner.model_config.head_dim
        self.batch_size = len(req_pool_indices)

        self.decode_wrapper = (
            decode_wrapper or self.model_runner.attn_backend.decode_wrapper
        )
        self.prefill_wrapper_ragged = (
            self.model_runner.attn_backend.prefill_wrapper_ragged
        )
        self.prefill_wrapper_paged = (
            self.model_runner.attn_backend.prefill_wrapper_paged
        )

        self.kv_last_page_len = torch.ones(
            (self.batch_size,), dtype=torch.int32, device="cuda"
        )

    def _init_indices_no_sliding_window(self):
        if self.use_ragged:
            paged_kernel_lens = self.prefix_lens
        else:
            paged_kernel_lens = self.seq_lens

        self.kv_indptr = torch.zeros(
            (self.batch_size + 1,), dtype=torch.int32, device="cuda"
        )
        self.kv_indptr[1:] = torch.cumsum(paged_kernel_lens, dim=0)
        self.kv_indices = torch.empty(
            self.kv_indptr[-1], dtype=torch.int32, device="cuda"
        )

        create_flashinfer_kv_indices_triton[(self.batch_size,)](
            self.model_runner.req_to_token_pool.req_to_token,
            self.req_pool_indices,
            paged_kernel_lens,
            self.kv_indptr,
            None,
            self.kv_indices,
            self.model_runner.req_to_token_pool.req_to_token.size(1),
        )

    def _init_indices_sliding_window(self, wrapper_id):
        if wrapper_id == 0:
            # window attention use paged only
            if self.forward_mode.is_decode():
                paged_kernel_lens = torch.minimum(
                    self.seq_lens,
                    torch.tensor(self.model_runner.sliding_window_size + 1),
                )
            else:
                paged_kernel_lens = torch.minimum(
                    self.seq_lens,
                    torch.tensor(self.model_runner.sliding_window_size)
                    + self.seq_lens
                    - self.prefix_lens,
                )
        else:
            # full attention
            paged_kernel_lens = self.seq_lens

        kv_start_idx = self.seq_lens - paged_kernel_lens
        self.kv_indptr = torch.zeros(
            (self.batch_size + 1,), dtype=torch.int32, device="cuda"
        )
        self.kv_indptr[1:] = torch.cumsum(paged_kernel_lens, dim=0)
        self.kv_indices = torch.empty(
            self.kv_indptr[-1], dtype=torch.int32, device="cuda"
        )
        create_flashinfer_kv_indices_triton[(self.batch_size,)](
            self.model_runner.req_to_token_pool.req_to_token,
            self.req_pool_indices,
            paged_kernel_lens,
            self.kv_indptr,
            kv_start_idx,
            self.kv_indices,
            self.model_runner.req_to_token_pool.req_to_token.size(1),
        )

    def _update_decode_indices(self, decode_wrapper):
        decode_wrapper.end_forward()
        decode_wrapper.begin_forward(
            self.kv_indptr,
            self.kv_indices,
            self.kv_last_page_len,
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            1,
            data_type=self.model_runner.kv_cache_dtype,
            q_data_type=self.model_runner.dtype,
        )

    def _update_extend_indices(self, ragged_wrapper, paged_wrapper):
        # extend part
        qo_indptr = torch.zeros(
            (self.batch_size + 1,), dtype=torch.int32, device="cuda"
        )
        qo_indptr[1:] = torch.cumsum(self.seq_lens - self.prefix_lens, dim=0)

        if self.use_ragged:
            ragged_wrapper.end_forward()
            ragged_wrapper.begin_forward(
                qo_indptr,
                qo_indptr,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
            )

        # cached part
        paged_wrapper.end_forward()
        paged_wrapper.begin_forward(
            qo_indptr,
            self.kv_indptr,
            self.kv_indices,
            self.kv_last_page_len,
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            1,
        )

    def update_indices_no_sliding_window(self):
        self._init_indices_no_sliding_window()

        if self.forward_mode.is_decode():
            self._update_decode_indices(self.decode_wrapper)
        else:
            self._update_extend_indices(
                self.prefill_wrapper_ragged,
                self.prefill_wrapper_paged,
            )

    def update_indices_sliding_window(self):
        assert self.use_ragged is False

        for wrapper_id in range(2):
            self._init_indices_sliding_window(wrapper_id)
            if self.forward_mode.is_decode():
                self._update_decode_indices(self.decode_wrapper[wrapper_id])
            else:
                self._update_extend_indices(
                    None,
                    self.prefill_wrapper_paged[wrapper_id],
                )


def update_flashinfer_indices(
    forward_mode,
    model_runner,
    req_pool_indices,
    seq_lens,
    prefix_lens,
    decode_wrapper=None,
    use_ragged=False,
):
    updater = FlashinferUpdater(
        forward_mode,
        model_runner,
        req_pool_indices,
        seq_lens,
        prefix_lens,
        decode_wrapper,
        use_ragged,
    )

    if model_runner.sliding_window_size is None:
        updater.update_indices_no_sliding_window()
    else:
        updater.update_indices_sliding_window()


# torchao configs


def torchao_quantize_param_data(param: torch.Tensor, torchao_config: str):
    """Quantize a Tensor with torchao quantization specified by torchao_config

    Args:
       `param`: weight parameter of the linear module
       `torchao_config`: type of quantization and their arguments we want to use to
        quantize the Tensor, e.g. int4wo-128 means int4 weight only quantization with group_size
        128
    """
    # Lazy import to suppress some warnings
    from torchao.quantization import (
        int4_weight_only,
        int8_dynamic_activation_int8_weight,
        int8_weight_only,
        quantize_,
    )

    dummy_linear = torch.nn.Linear(param.shape[1], param.shape[0], bias=False)
    dummy_linear.weight = param
    if "int8wo" in torchao_config:
        quantize_(dummy_linear, int8_weight_only())
    elif "int8dq" in torchao_config:
        quantize_(dummy_linear, int8_dynamic_activation_int8_weight())
    elif "int4wo" in torchao_config:
        group_size = int(torchao_config.split("-")[-1])
        assert group_size in [
            32,
            64,
            128,
            256,
        ], f"int4wo groupsize needs to be one of [32, 64, 128, 256] but got {group_size}"
        quantize_(dummy_linear, int4_weight_only(group_size=group_size))
    elif "fp8wo" in torchao_config:
        from torchao.quantization import float8_weight_only

        # this requires newer hardware
        # [rank0]: AssertionError: fp8e4nv data type is not supported on CUDA arch < 89
        quantize_(dummy_linear, float8_weight_only())
    return dummy_linear.weight


def apply_torchao_config_(
    self: torch.nn.Module,
    params_dict: Dict[str, torch.Tensor],
    param_suffixes: Set[str],
) -> None:
    """A util function used for quantizing the weight parameters after they are loaded if
       self.torchao_config is specified

    Args:
      `self`: the model we want to quantize
      `params_dict`: dictionary mapping from param_name to the parameter Tensor
      `param_suffixes`: a set of suffixes, we'll quantize the Tensor matching these suffixes

    Returns:
       None, the `params_dict` is modified inplace and the weights of `self` model are quantized
    """
    if self.torchao_config:
        for param_suffix in param_suffixes:
            for name in params_dict:
                param = params_dict[name]
                if param_suffix in name and param.ndim == 2:
                    params_dict[name] = torchao_quantize_param_data(
                        param, self.torchao_config
                    )
        self.load_state_dict(params_dict, assign=True)
