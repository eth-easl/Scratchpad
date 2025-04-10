import enum
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Union,
)
import torch
from pathlib import Path
from transformers import PretrainedConfig, AutoConfig
from scratchpad.utils import (
    logger,
    get_hf_text_config,
    print_warning_once,
    get_hf_image_processor_config,
)
from scratchpad.config.modality_config import MultiModalConfig
import huggingface_hub
from huggingface_hub import file_exists, try_to_load_from_cache
from transformers.utils import CONFIG_NAME as HF_CONFIG_NAME

from .utils import (
    _get_and_verify_dtype,
    _get_and_verify_max_len,
    get_served_model_name,
    is_multimodal_model,
)


class ConfigFormat(str, enum.Enum):
    AUTO = "auto"
    HF = "hf"
    MISTRAL = "mistral"


def file_or_path_exists(model: Union[str, Path], config_name, revision, token) -> bool:
    if Path(model).exists():
        return (Path(model) / config_name).is_file()

    # Offline mode support: Check if config file is cached already
    cached_filepath = try_to_load_from_cache(
        repo_id=model, filename=config_name, revision=revision
    )
    if isinstance(cached_filepath, str):
        # The config file exists in cache- we can continue trying to load
        return True

    # NB: file_exists will only check for the existence of the config file on
    # hf_hub. This will fail in offline mode.
    try:
        return file_exists(model, config_name, revision=revision, token=token)
    except huggingface_hub.errors.OfflineModeIsEnabled:
        # Don't raise in offline mode, all we know is that we don't have this
        # file cached.
        return False


def get_vllm_config(
    model: Union[str, Path],
    trust_remote_code: bool,
    revision: Optional[str] = None,
    code_revision: Optional[str] = None,
    rope_scaling: Optional[dict] = None,
    rope_theta: Optional[float] = None,
    config_format: ConfigFormat = ConfigFormat.AUTO,
    **kwargs,
) -> PretrainedConfig:
    # Separate model folder from file path for GGUF models

    if config_format == ConfigFormat.AUTO:
        if file_or_path_exists(
            model, HF_CONFIG_NAME, revision=revision, token=kwargs.get("token")
        ):
            config_format = ConfigFormat.HF
        else:
            # If we're in offline mode and found no valid config format, then
            # raise an offline mode error to indicate to the user that they
            # don't have files cached and may need to go online.
            # This is conveniently triggered by calling file_exists().
            raise ValueError(f"No supported config format found in {model}")

    if config_format == ConfigFormat.HF:
        config_dict, _ = PretrainedConfig.get_config_dict(
            model, revision=revision, code_revision=code_revision, **kwargs
        )

        # Use custom model class if it's in our registry
        model_type = config_dict.get("model_type")
        try:
            config = AutoConfig.from_pretrained(
                model,
                trust_remote_code=trust_remote_code,
                revision=revision,
                code_revision=code_revision,
                **kwargs,
            )
        except ValueError as e:
            if (
                not trust_remote_code
                and "requires you to execute the configuration file" in str(e)
            ):
                err_msg = (
                    "Failed to load the model config. If the model "
                    "is a custom model not yet available in the "
                    "HuggingFace transformers library, consider setting "
                    "`trust_remote_code=True` in LLM or using the "
                    "`--trust-remote-code` flag in the CLI."
                )
                raise RuntimeError(err_msg) from e
            else:
                raise e

    else:
        raise ValueError(f"Unsupported config format: {config_format}")

    # Special architecture mapping check for GGUF models

    for key, value in [
        ("rope_scaling", rope_scaling),
        ("rope_theta", rope_theta),
    ]:
        if value is not None:
            logger.info(
                "Updating %s from %r to %r",
                key,
                getattr(config, key, None),
                value,
            )
            config.update({key: value})

    return config


class ModelConfig:
    """Configuration for the model.

    Args:
        model: Name or path of the huggingface model to use.
            It is also used as the content for `model_name` tag in metrics
            output when `served_model_name` is not specified.
        tokenizer: Name or path of the huggingface tokenizer to use.
        tokenizer_mode: Tokenizer mode. "auto" will use the fast tokenizer if
            available, "slow" will always use the slow tokenizer, and
            "mistral" will always use the tokenizer from `mistral_common`.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        dtype: Data type for model weights and activations. The "auto" option
            will use FP16 precision for FP32 and FP16 models, and BF16 precision
            for BF16 models.
        seed: Random seed for reproducibility.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id. If unspecified, will use the default
            version.
        code_revision: The specific revision to use for the model code on
            Hugging Face Hub. It can be a branch name, a tag name, or a
            commit id. If unspecified, will use the default version.
        rope_scaling: Dictionary containing the scaling configuration for the
            RoPE embeddings. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum.
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id. If unspecified, will use
            the default version.
        max_model_len: Maximum length of a sequence (including prompt and
            output). If None, will be derived from the model.
        quantization: Quantization method that was used to quantize the model
            weights. If None, we assume the model weights are not quantized.
        quantization_param_path: Path to JSON file containing scaling factors.
            Used to load KV cache scaling factors into the model when KV cache
            type is FP8_E4M3 on ROCm (AMD GPU). In the future these will also
            be used to load activation and weight scaling factors when the
            model dtype is FP8_E4M3 on ROCm.
        enforce_eager: Whether to enforce eager execution. If True, we will
            disable CUDA graph and always execute the model in eager mode.
            If False, we will use CUDA graph and eager execution in hybrid.
            If None, the user did not specify, so default to False.
        max_context_len_to_capture: Maximum context len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode (DEPRECATED. Use max_seq_len_to_capture instead).
        max_seq_len_to_capture: Maximum sequence len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode. Additionally for encoder-decoder models, if the
            sequence length of the encoder input is larger than this, we fall
            back to the eager mode.
        disable_sliding_window: Whether to disable sliding window. If True,
            we will disable the sliding window functionality of the model.
            If the model does not support sliding window, this argument is
            ignored.
        skip_tokenizer_init: If true, skip initialization of tokenizer and
            detokenizer.
        served_model_name: The model name used in metrics tag `model_name`,
            matches the model name exposed via the APIs. If multiple model
            names provided, the first name will be used. If not specified,
            the model name will be the same as `model`.
        limit_mm_per_prompt: Maximum number of data instances per modality
            per prompt. Only applicable for multimodal models.
        override_neuron_config: Initialize non default neuron config or
            override default neuron config that are specific to Neuron devices,
            this argument will be used to configure the neuron config that
            can not be gathered from the vllm arguments.
        config_format: The config format which shall be loaded.
            Defaults to 'auto' which defaults to 'hf'.
        mm_processor_kwargs: Arguments to be forwarded to the model's processor
            for multi-modal data, e.g., image processor.
    """

    def __init__(
        self,
        model: str,
        tokenizer: str,
        tokenizer_mode: str,
        trust_remote_code: bool,
        dtype: Union[str, torch.dtype],
        seed: int,
        revision: Optional[str] = None,
        code_revision: Optional[str] = None,
        rope_scaling: Optional[dict] = None,
        rope_theta: Optional[float] = None,
        tokenizer_revision: Optional[str] = None,
        max_model_len: Optional[int] = None,
        spec_target_max_model_len: Optional[int] = None,
        quantization: Optional[str] = None,
        quantization_param_path: Optional[str] = None,
        enforce_eager: Optional[bool] = None,
        max_context_len_to_capture: Optional[int] = None,
        max_seq_len_to_capture: Optional[int] = None,
        max_logprobs: int = 20,
        disable_sliding_window: bool = False,
        skip_tokenizer_init: bool = False,
        served_model_name: Optional[Union[str, List[str]]] = None,
        limit_mm_per_prompt: Optional[Mapping[str, int]] = None,
        use_async_output_proc: bool = True,
        override_neuron_config: Optional[Dict[str, Any]] = None,
        config_format: ConfigFormat = ConfigFormat.AUTO,
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_mode = tokenizer_mode
        self.trust_remote_code = trust_remote_code
        self.seed = seed
        self.revision = revision
        self.code_revision = code_revision
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta
        # The tokenizer version is consistent with the model version by default.
        if tokenizer_revision is None:
            self.tokenizer_revision = revision
        else:
            self.tokenizer_revision = tokenizer_revision
        self.quantization = quantization
        self.quantization_param_path = quantization_param_path
        self.enforce_eager = enforce_eager
        if max_context_len_to_capture is not None:
            raise ValueError(
                "`max_context_len_to_capture` is deprecated. "
                "Use `max_seq_len_to_capture` instead."
            )
        self.max_seq_len_to_capture = max_seq_len_to_capture
        self.max_logprobs = max_logprobs
        self.disable_sliding_window = disable_sliding_window
        self.skip_tokenizer_init = skip_tokenizer_init

        self.hf_config = get_vllm_config(
            self.model,
            trust_remote_code,
            revision,
            code_revision,
            rope_scaling,
            rope_theta,
            config_format,
        )
        self.hf_text_config = get_hf_text_config(self.hf_config)
        self.hf_image_processor_config = get_hf_image_processor_config(
            self.model, revision
        )
        self.dtype = _get_and_verify_dtype(self.hf_text_config, dtype)
        self.use_async_output_proc = use_async_output_proc
        self.mm_processor_kwargs = mm_processor_kwargs

        # Set enforce_eager to False if the value is unset.
        if self.enforce_eager is None:
            self.enforce_eager = False

        if (
            not self.disable_sliding_window
            and self.hf_text_config.model_type == "gemma2"
            and self.hf_text_config.sliding_window is not None
        ):
            print_warning_once(
                "Gemma 2 uses sliding window attention for every odd layer, "
                "which is currently not supported by vLLM. Disabling sliding "
                "window and capping the max length to the sliding window size "
                f"({self.hf_text_config.sliding_window})."
            )
            self.disable_sliding_window = True

        self.max_model_len = _get_and_verify_max_len(
            hf_config=self.hf_text_config,
            max_model_len=max_model_len,
            disable_sliding_window=self.disable_sliding_window,
            sliding_window_len=self.get_hf_config_sliding_window(),
            spec_target_max_model_len=spec_target_max_model_len,
        )
        self.served_model_name = get_served_model_name(model, served_model_name)
        self.multimodal_config = self._init_multimodal_config(limit_mm_per_prompt)
        if not self.skip_tokenizer_init:
            self._verify_tokenizer_mode()

        self.override_neuron_config = None
        self._verify_cuda_graph()
        self._verify_bnb_config()

    def _init_multimodal_config(
        self, limit_mm_per_prompt: Optional[Mapping[str, int]]
    ) -> Optional["MultiModalConfig"]:
        architectures = getattr(self.hf_config, "architectures", [])
        if is_multimodal_model(architectures):
            return MultiModalConfig(limit_per_prompt=limit_mm_per_prompt or {})
        else:
            if limit_mm_per_prompt:
                raise ValueError(
                    "limit_mm_per_prompt is only supported for multimodal " "models."
                )
            return None

    def _verify_tokenizer_mode(self) -> None:
        tokenizer_mode = self.tokenizer_mode.lower()
        if tokenizer_mode not in ["auto", "slow", "mistral"]:
            raise ValueError(
                f"Unknown tokenizer mode: {self.tokenizer_mode}. Must be "
                "either 'auto', 'slow' or 'mistral'."
            )
        self.tokenizer_mode = tokenizer_mode

    def _parse_quant_hf_config(self):
        quant_cfg = getattr(self.hf_config, "quantization_config", None)
        if quant_cfg is None:
            # compressed-tensors uses a "compression_config" key
            quant_cfg = getattr(self.hf_config, "compression_config", None)
        return quant_cfg

    # def _verify_quantization(self) -> None:
    #     supported_quantization = [*QUANTIZATION_METHODS]
    #     rocm_supported_quantization = [
    #         "awq",
    #         "gptq",
    #         "fp8",
    #         "compressed_tensors",
    #         "compressed-tensors",
    #         "fbgemm_fp8",
    #     ]
    #     optimized_quantization_methods = [
    #         "fp8",
    #         "marlin",
    #         "modelopt",
    #         "gptq_marlin_24",
    #         "gptq_marlin",
    #         "awq_marlin",
    #         "fbgemm_fp8",
    #         "compressed_tensors",
    #         "compressed-tensors",
    #         "experts_int8",
    #     ]
    #     tpu_supported_quantization = ["tpu_int8"]
    #     neuron_supported_quantization = ["neuron_quant"]
    #     if self.quantization is not None:
    #         self.quantization = self.quantization.lower()

    #     # Parse quantization method from the HF model config, if available.
    #     quant_cfg = self._parse_quant_hf_config()

    #     if quant_cfg is not None:
    #         quant_method = quant_cfg.get("quant_method", "").lower()

    #         # Detect which checkpoint is it
    #         for _, method in QUANTIZATION_METHODS.items():
    #             quantization_override = method.override_quantization_method(
    #                 quant_cfg, self.quantization
    #             )
    #             if quantization_override:
    #                 quant_method = quantization_override
    #                 self.quantization = quantization_override
    #                 break

    #         # Verify quantization configurations.
    #         if self.quantization is None:
    #             self.quantization = quant_method
    #         elif self.quantization != quant_method:
    #             raise ValueError(
    #                 "Quantization method specified in the model config "
    #                 f"({quant_method}) does not match the quantization "
    #                 f"method specified in the `quantization` argument "
    #                 f"({self.quantization})."
    #             )

    #     if self.quantization is not None:
    #         if self.quantization not in supported_quantization:
    #             raise ValueError(
    #                 f"Unknown quantization method: {self.quantization}. Must "
    #                 f"be one of {supported_quantization}."
    #             )
    #         if is_hip() and self.quantization not in rocm_supported_quantization:
    #             raise ValueError(
    #                 f"{self.quantization} quantization is currently not "
    #                 f"supported in ROCm."
    #             )
    #         if (
    #             current_platform.is_tpu()
    #             and self.quantization not in tpu_supported_quantization
    #         ):
    #             raise ValueError(
    #                 f"{self.quantization} quantization is currently not "
    #                 f"supported in TPU Backend."
    #             )
    #         if self.quantization not in optimized_quantization_methods:
    #             logger.warning(
    #                 "%s quantization is not fully "
    #                 "optimized yet. The speed can be slower than "
    #                 "non-quantized models.",
    #                 self.quantization,
    #             )
    #         if self.quantization == "awq" and is_hip() and not envs.VLLM_USE_TRITON_AWQ:
    #             logger.warning(
    #                 "Using AWQ quantization with ROCm, but VLLM_USE_TRITON_AWQ"
    #                 " is not set, enabling VLLM_USE_TRITON_AWQ."
    #             )
    #             envs.VLLM_USE_TRITON_AWQ = True
    #         if is_neuron() and self.quantization not in neuron_supported_quantization:
    #             raise ValueError(
    #                 f"{self.quantization} quantization is currently not "
    #                 f"supported in Neuron Backend."
    #             )

    def _verify_cuda_graph(self) -> None:
        if self.max_seq_len_to_capture is None:
            self.max_seq_len_to_capture = self.max_model_len
        self.max_seq_len_to_capture = min(
            self.max_seq_len_to_capture, self.max_model_len
        )

    def _verify_bnb_config(self) -> None:
        """
        The current version of bitsandbytes (0.44.0) with 8-bit models does not
        yet support CUDA graph.
        """
        is_bitsandbytes = self.quantization == "bitsandbytes"
        has_quantization_config = (
            getattr(self.hf_config, "quantization_config", None) is not None
        )
        is_8bit = (
            self.hf_config.quantization_config.get("load_in_8bit", False)
            if has_quantization_config
            else False
        )
        if all(
            [
                is_bitsandbytes,
                has_quantization_config,
                is_8bit,
                not self.enforce_eager,
            ]
        ):
            logger.warning(
                "CUDA graph is not supported on BitAndBytes 8bit yet, "
                "fallback to the eager mode."
            )
            self.enforce_eager = True

    def get_hf_config_sliding_window(self) -> Optional[int]:
        """Get the sliding window size, or None if disabled."""

        # Some models, like Qwen2 and Qwen1.5, use `use_sliding_window` in
        # addition to sliding window size. We check if that field is present
        # and if it's False, return None.
        if (
            hasattr(self.hf_text_config, "use_sliding_window")
            and not self.hf_text_config.use_sliding_window
        ):
            return None
        return getattr(self.hf_text_config, "sliding_window", None)

    def get_sliding_window(self) -> Optional[int]:
        """Get the sliding window size, or None if disabled."""
        # If user disables sliding window, return None.
        if self.disable_sliding_window:
            return None
        # Otherwise get the value from the hf config.
        return self.get_hf_config_sliding_window()

    def get_vocab_size(self) -> int:
        return self.hf_text_config.vocab_size

    def get_hidden_size(self) -> int:
        return self.hf_text_config.hidden_size

    def get_head_size(self) -> int:
        # TODO remove hard code
        if (
            hasattr(self.hf_text_config, "model_type")
            and self.hf_text_config.model_type == "deepseek_v2"
        ):
            # FlashAttention supports only head_size 32, 64, 128, 256,
            # we need to pad head_size 192 to 256
            return 256
        if hasattr(self.hf_text_config, "head_dim"):
            return self.hf_text_config.head_dim
        # FIXME(woosuk): This may not be true for all models.
        return (
            self.hf_text_config.hidden_size // self.hf_text_config.num_attention_heads
        )

    @property
    def is_encoder_decoder_model(self) -> bool:
        """Extract the HF encoder/decoder model flag."""
        return getattr(self.hf_config, "is_encoder_decoder", False) or (
            (
                hasattr(self.hf_config, "text_config")
                and getattr(self.hf_config.text_config, "is_encoder_decoder", False)
            )
        )

    @property
    def is_multimodal_model(self) -> bool:
        return self.multimodal_config is not None
