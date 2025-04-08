import torch
from transformers import PretrainedConfig
from scratchpad.utils import logger, envs
from typing import Union, Optional, List, Any

_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.float16,
    "float16": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def _get_and_verify_dtype(
    config: PretrainedConfig,
    dtype: Union[str, torch.dtype],
) -> torch.dtype:
    # NOTE: getattr(config, "torch_dtype", torch.float32) is not correct
    # because config.torch_dtype can be None.
    config_dtype = getattr(config, "torch_dtype", None)
    if config_dtype is None:
        config_dtype = torch.float32

    if isinstance(dtype, str):
        dtype = dtype.lower()
        if dtype == "auto":
            if config_dtype == torch.float32:
                if config.model_type == "gemma2":
                    logger.info(
                        "For Gemma 2, we downcast float32 to bfloat16 instead "
                        "of float16 by default. Please specify `dtype` if you "
                        "want to use float16."
                    )
                    torch_dtype = torch.bfloat16
                else:
                    # Following the common practice, we use float16 for float32
                    # models.
                    torch_dtype = torch.float16
            else:
                torch_dtype = config_dtype
        else:
            if dtype not in _STR_DTYPE_TO_TORCH_DTYPE:
                raise ValueError(f"Unknown dtype: {dtype}")
            torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE[dtype]
    elif isinstance(dtype, torch.dtype):
        torch_dtype = dtype
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    # Verify the dtype.
    if torch_dtype != config_dtype:
        if torch_dtype == torch.float32:
            # Upcasting to float32 is allowed.
            logger.info(f"Upcasting {config_dtype} to {torch_dtype}.")
            pass
        elif config_dtype == torch.float32:
            # Downcasting from float32 to float16 or bfloat16 is allowed.
            logger.info(f"Downcasting {config_dtype} to {torch_dtype}.")
            pass
        else:
            # Casting between float16 and bfloat16 is allowed with a warning.
            logger.warning(f"Casting {config_dtype} to {torch_dtype}.")

    return torch_dtype


def get_min_sliding_window(sliding_window: Union[int, list[Optional[int]]]) -> int:
    if isinstance(sliding_window, list):
        return min(s for s in sliding_window if s is not None)

    return sliding_window


def _get_and_verify_max_len(
    hf_config: PretrainedConfig,
    max_model_len: Optional[int],
    disable_sliding_window: bool,
    sliding_window_len: Optional[Union[int, List[Optional[int]]]],
    spec_target_max_model_len: Optional[int] = None,
    encoder_config: Optional[Any] = None,
) -> int:
    """Get and verify the model's maximum length."""
    derived_max_model_len = float("inf")
    possible_keys = [
        # OPT
        "max_position_embeddings",
        # GPT-2
        "n_positions",
        # MPT
        "max_seq_len",
        # ChatGLM2
        "seq_length",
        # Command-R
        "model_max_length",
        # Whisper
        "max_target_positions",
        # Others
        "max_sequence_length",
        "max_seq_length",
        "seq_len",
    ]
    # Choose the smallest "max_length" from the possible keys.
    max_len_key = None
    for key in possible_keys:
        max_len = getattr(hf_config, key, None)
        if max_len is not None:
            max_len_key = key if max_len < derived_max_model_len else max_len_key
            derived_max_model_len = min(derived_max_model_len, max_len)

    # If sliding window is manually disabled, max_length should be less
    # than the sliding window length in the model config.
    if disable_sliding_window and sliding_window_len is not None:

        sliding_window_len_min = get_min_sliding_window(sliding_window_len)
        max_len_key = (
            "sliding_window"
            if sliding_window_len_min < derived_max_model_len
            else max_len_key
        )
        derived_max_model_len = min(derived_max_model_len, sliding_window_len_min)

    # If none of the keys were found in the config, use a default and
    # log a warning.
    if derived_max_model_len == float("inf"):
        if max_model_len is not None:
            # If max_model_len is specified, we use it.
            return max_model_len

        if spec_target_max_model_len is not None:
            # If this is a speculative draft model, we use the max model len
            # from the target model.
            return spec_target_max_model_len

        default_max_len = 2048
        logger.warning(
            "The model's config.json does not contain any of the following "
            "keys to determine the original maximum length of the model: "
            "%s. Assuming the model's maximum length is %d.",
            possible_keys,
            default_max_len,
        )
        derived_max_model_len = default_max_len

    rope_scaling = getattr(hf_config, "rope_scaling", None)
    if rope_scaling is not None:
        # No need to consider "type" key because of patch_rope_scaling when
        # loading HF config
        rope_type = rope_scaling["rope_type"]

        if rope_type not in ("su", "longrope", "llama3"):
            if disable_sliding_window:
                # TODO(robertgshaw): Find a model that supports rope_scaling
                # with sliding window to see if this case should be allowed.
                raise NotImplementedError(
                    "Disabling sliding window is not supported for models "
                    "with rope_scaling. Please raise an issue so we can "
                    "investigate."
                )

            # NOTE: rope_type == "default" does not define factor
            # https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/modeling_rope_utils.py
            scaling_factor = rope_scaling.get("factor", 1.0)
            if rope_type == "yarn":
                derived_max_model_len = rope_scaling["original_max_position_embeddings"]
            derived_max_model_len *= scaling_factor

    if encoder_config and "max_seq_length" in encoder_config:
        derived_max_model_len = encoder_config["max_seq_length"]

    # If the user specified a max length, make sure it is smaller than the
    # derived length from the HF model config.
    if max_model_len is None:
        max_model_len = int(derived_max_model_len)
    elif max_model_len > derived_max_model_len:
        # Some models might have a separate key for specifying model_max_length
        # that will be bigger than derived_max_model_len. We compare user input
        # with model_max_length and allow this override when it's smaller.
        model_max_length = getattr(hf_config, "model_max_length", None)
        if model_max_length is not None and max_model_len <= model_max_length:
            if disable_sliding_window:
                # TODO(robertgshaw): Find a model that has model_max_length
                # with sliding window to see if this case should be allowed.
                raise NotImplementedError(
                    "Disabling sliding window is not supported for models "
                    "model_max_length in the config. Please raise an issue "
                    "so we can investigate."
                )
        else:
            msg = (
                f"User-specified max_model_len ({max_model_len}) is greater "
                f"than the derived max_model_len ({max_len_key}="
                f"{derived_max_model_len} or model_max_length="
                f"{model_max_length} in model's config.json). This may lead "
                "to incorrect model outputs or CUDA errors."
            )
            if envs.VLLM_ALLOW_LONG_MAX_MODEL_LEN:
                logger.warning(
                    "%s Make sure the value is correct and within the "
                    "model context size.",
                    msg,
                )
            else:
                raise ValueError(
                    f"{msg} To allow overriding this maximum, set "
                    "the env var SP_ALLOW_LONG_MAX_MODEL_LEN=1"
                )
    return int(max_model_len)


def get_served_model_name(
    model: str, served_model_name: Optional[Union[str, List[str]]]
):
    """
    If the input is a non-empty list, the first model_name in
    `served_model_name` is taken.
    If the input is a non-empty string, it is used directly.
    For cases where the input is either an empty string or an
    empty list, the fallback is to use `self.model`.
    """
    if not served_model_name:
        return model
    if isinstance(served_model_name, list):
        return served_model_name[0]
    return served_model_name


multimodal_model_archs = [
    "DeepseekVL2ForCausalLM",
    "Gemma3ForConditionalGeneration",
    "Grok1VForCausalLM",
    "Grok1AForCausalLM",
    "LlavaLlamaForCausalLM",
    "LlavaMistralForCausalLM",
    "LlavaQwenForCausalLM",
    "LlavaVidForCausalLM",
    "MiniCPMO",
    "MiniCPMV",
    "MultiModalityCausalLM",
    "MllamaForConditionalGeneration",
    "Qwen2VLForConditionalGeneration",
    "Qwen2_5_VLForConditionalGeneration",
    "CLIPModel",
]


def is_multimodal_model(model_architectures: List[str]):
    if any(
        multi_model_arch in model_architectures
        for multi_model_arch in multimodal_model_archs
    ):
        return True
    else:
        return False
