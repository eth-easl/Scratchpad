import copy
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from scratchpad.scheduler.schedule_batch import BaseFinishReason
from scratchpad.sampling.sampling_params import SamplingParams
from enum import Enum


@dataclass
class SessionParams:
    id: Optional[str] = None
    rid: Optional[str] = None
    offset: Optional[int] = None
    replace: Optional[bool] = None


@dataclass
class GenerateReqInput:
    # The input prompt. It can be a single prompt or a batch of prompts.
    text: Optional[Union[List[str], str]] = None
    # The token ids for text; one can specify either text or input_ids
    input_ids: Optional[Union[List[List[int]], List[int]]] = None
    # The embeddings for input_ids; one can specify either text or input_ids or input_embeds.
    input_embeds: Optional[Union[List[List[List[float]]], List[List[float]]]] = None
    # The image input. It can be a file name, a url, or base64 encoded string.
    # See also python/sglang/srt/utils.py:load_image.
    image_data: Optional[Union[List[str], str]] = None
    # The sampling_params. See descriptions below.
    sampling_params: Optional[Union[List[Dict], Dict]] = None
    # The request id.
    rid: Optional[Union[List[str], str]] = None
    # Whether to return logprobs.
    return_logprob: Optional[Union[List[bool], bool]] = None
    # If return logprobs, the start location in the prompt for returning logprobs.
    # By default, this value is "-1", which means it will only return logprobs for output tokens.
    logprob_start_len: Optional[Union[List[int], int]] = None
    # If return logprobs, the number of top logprobs to return at each position.
    top_logprobs_num: Optional[Union[List[int], int]] = None
    # Whether to detokenize tokens in text in the returned logprobs.
    return_text_in_logprobs: bool = False
    # Whether to stream output.
    stream: bool = False
    # The modalities of the image data [image, multi-images, video]
    modalities: Optional[List[str]] = None
    # Topping related
    topping_path: Optional[Union[List[Optional[str]], Optional[str]]] = None

    # Session info for continual prompting
    session_params: Optional[Union[List[Dict], Dict]] = None

    def normalize_batch_and_arguments(self):
        if (
            self.text is None and self.input_ids is None and self.input_embeds is None
        ) or (
            self.text is not None
            and self.input_ids is not None
            and self.input_embeds is not None
        ):
            raise ValueError(
                "Either text, input_ids or input_embeds should be provided."
            )

        # Derive the batch size
        if self.text is not None:
            if isinstance(self.text, str):
                self.is_single = True
                self.batch_size = 1
            else:
                self.is_single = False
                self.batch_size = len(self.text)
            self.input_embeds = None
        elif self.input_ids is not None:
            if isinstance(self.input_ids[0], int):
                self.is_single = True
                self.batch_size = 1
            else:
                self.is_single = False
                self.batch_size = len(self.input_ids)
            self.input_embeds = None
        else:
            if isinstance(self.input_embeds[0][0], float):
                self.is_single = True
                self.batch_size = 1
            else:
                self.batch_size = len(self.input_embeds)

        # Handle parallel sampling
        # When parallel sampling is used, we always treat the input as a batch.
        if self.sampling_params is None:
            self.parallel_sample_num = 1
        elif isinstance(self.sampling_params, dict):
            self.parallel_sample_num = self.sampling_params.get("n", 1)
        else:  # isinstance(self.sampling_params, list):
            self.parallel_sample_num = self.sampling_params[0].get("n", 1)
            assert all(
                self.parallel_sample_num == sampling_params.get("n", 1)
                for sampling_params in self.sampling_params
            ), "The parallel_sample_num should be the same for all samples in sample params."

        if self.parallel_sample_num > 1 and self.is_single:
            self.is_single = False
            if self.text is not None:
                self.text = [self.text]
            if self.input_ids is not None:
                self.input_ids = [self.input_ids]

        # Fill in default arguments
        if self.is_single:
            if self.sampling_params is None:
                self.sampling_params = {}
            if self.rid is None:
                self.rid = uuid.uuid4().hex
            if self.return_logprob is None:
                self.return_logprob = False
            if self.logprob_start_len is None:
                self.logprob_start_len = -1
            if self.top_logprobs_num is None:
                self.top_logprobs_num = 0
        else:
            if self.parallel_sample_num == 1:
                num = self.batch_size
            else:
                # Expand parallel_sample_num
                num = self.batch_size * self.parallel_sample_num

            if self.image_data is None:
                self.image_data = [None] * num
            elif not isinstance(self.image_data, list):
                self.image_data = [self.image_data] * num
            elif isinstance(self.image_data, list):
                pass

            if self.sampling_params is None:
                self.sampling_params = [{}] * num
            elif not isinstance(self.sampling_params, list):
                self.sampling_params = [self.sampling_params] * num

            if self.rid is None:
                self.rid = [uuid.uuid4().hex for _ in range(num)]
            else:
                assert isinstance(self.rid, list), "The rid should be a list."

            if self.return_logprob is None:
                self.return_logprob = [False] * num
            elif not isinstance(self.return_logprob, list):
                self.return_logprob = [self.return_logprob] * num
            else:
                assert self.parallel_sample_num == 1

            if self.logprob_start_len is None:
                self.logprob_start_len = [-1] * num
            elif not isinstance(self.logprob_start_len, list):
                self.logprob_start_len = [self.logprob_start_len] * num
            else:
                assert self.parallel_sample_num == 1

            if self.top_logprobs_num is None:
                self.top_logprobs_num = [0] * num
            elif not isinstance(self.top_logprobs_num, list):
                self.top_logprobs_num = [self.top_logprobs_num] * num
            else:
                assert self.parallel_sample_num == 1

    def regenerate_rid(self):
        self.rid = uuid.uuid4().hex
        return self.rid

    def __getitem__(self, i):
        return GenerateReqInput(
            text=self.text[i] if self.text is not None else None,
            input_ids=self.input_ids[i] if self.input_ids is not None else None,
            image_data=self.image_data[i],
            sampling_params=self.sampling_params[i],
            rid=self.rid[i],
            return_logprob=self.return_logprob[i],
            logprob_start_len=self.logprob_start_len[i],
            top_logprobs_num=self.top_logprobs_num[i],
            return_text_in_logprobs=self.return_text_in_logprobs,
            stream=self.stream,
            modalities=self.modalities[i] if self.modalities else None,
            topping_path=self.topping_path[i]
            if self.topping_path is not None
            else None,
        )


@dataclass
class MemoryPoolControlReqInput:
    delta: int
    is_expand: bool = False
    is_shrink: bool = False

    def __post_init__(self):
        if self.is_expand and self.is_shrink:
            raise ValueError("is_expand and is_shrink cannot be True at the same time.")


@dataclass
class TokenizedGenerateReqInput:
    # The request id
    rid: str
    # The input text
    input_text: str
    # The input token ids
    input_ids: List[int]
    # The image input
    image_inputs: dict
    # The sampling parameters
    sampling_params: SamplingParams
    # Whether to return the logprobs
    return_logprob: bool
    # If return logprobs, the start location in the prompt for returning logprobs.
    logprob_start_len: int
    # If return logprobs, the number of top logprobs to return at each position.
    top_logprobs_num: int
    # Whether to stream output
    stream: bool

    # LoRA related
    topping_path: Optional[str] = None  # None means just use the base model
    # The input embeds
    input_embeds: Optional[Union[List[List[List[float]]], List[List[float]]]] = None

    # Session info for continual prompting
    session_params: Optional[SessionParams] = None


@dataclass
class EmbeddingReqInput:
    # The input prompt. It can be a single prompt or a batch of prompts.
    text: Optional[Union[List[str], str]] = None
    # The token ids for text; one can either specify text or input_ids.
    input_ids: Optional[Union[List[List[int]], List[int]]] = None
    # The request id.
    rid: Optional[Union[List[str], str]] = None
    # Dummy sampling params for compatibility
    sampling_params: Union[List[Dict], Dict] = None
    input_embeds: Optional[Union[List[List[List[float]]], List[List[float]]]] = None
    is_single: bool = True

    def post_init(self):
        if (self.text is None and self.input_ids is None) or (
            self.text is not None and self.input_ids is not None
        ):
            raise ValueError("Either text or input_ids should be provided.")

        if self.text is not None:
            is_single = isinstance(self.text, str)
        else:
            is_single = isinstance(self.input_ids[0], int)
        self.is_single = is_single

        if is_single:
            if self.rid is None:
                self.rid = uuid.uuid4().hex
            if self.sampling_params is None:
                self.sampling_params = {}
            self.sampling_params["max_new_tokens"] = 1
        else:
            # support select operation
            self.batch_size = (
                len(self.text) if self.text is not None else len(self.input_ids)
            )
            if self.rid is None:
                self.rid = [uuid.uuid4().hex for _ in range(self.batch_size)]
            else:
                if not isinstance(self.rid, list):
                    raise ValueError("The rid should be a list.")
            if self.sampling_params is None:
                self.sampling_params = [{}] * self.batch_size
            for i in range(self.batch_size):
                self.sampling_params[i]["max_new_tokens"] = 1

    def normalize_batch_and_arguments(self):
        if (self.text is None and self.input_ids is None) or (
            self.text is not None and self.input_ids is not None
        ):
            raise ValueError("Either text or input_ids should be provided.")

        # Derive the batch size
        if self.text is not None:
            if isinstance(self.text, str):
                self.is_single = True
                self.batch_size = 1
            else:
                self.is_single = False
                self.batch_size = len(self.text)
        else:
            if isinstance(self.input_ids[0], int):
                self.is_single = True
                self.batch_size = 1
            else:
                self.is_single = False
                self.batch_size = len(self.input_ids)

        # Fill in default arguments
        if self.is_single:
            if self.rid is None:
                self.rid = uuid.uuid4().hex
            if self.sampling_params is None:
                self.sampling_params = {}
            self.sampling_params["max_new_tokens"] = 0
        else:
            if self.rid is None:
                self.rid = [uuid.uuid4().hex for _ in range(self.batch_size)]
            else:
                assert isinstance(self.rid, list), "The rid should be a list."

            if self.sampling_params is None:
                self.sampling_params = [{}] * self.batch_size
            for i in range(self.batch_size):
                self.sampling_params[i]["max_new_tokens"] = 0

    def regenerate_rid(self):
        self.rid = uuid.uuid4().hex
        return self.rid

    def __getitem__(self, i):
        return EmbeddingReqInput(
            text=self.text[i] if self.text is not None else None,
            input_ids=self.input_ids[i] if self.input_ids is not None else None,
            sampling_params=self.sampling_params[i],
            rid=self.rid[i],
        )


@dataclass
class TokenizedEmbeddingReqInput:
    # The request id
    rid: str
    # The input text
    input_text: str
    # The input token ids
    input_ids: List[int]
    # Dummy sampling params for compatibility
    sampling_params: SamplingParams


@dataclass
class BatchTokenIDOut:
    # The request id
    rids: List[str]
    # The finish reason
    finished_reasons: List[BaseFinishReason]
    # For incremental decoding
    # The version id to sync decode status with in detokenizer_manager
    vids: List[int]
    decoded_texts: List[str]
    decode_ids: List[int]
    read_offsets: List[int]
    # Only used when `--skip-tokenizer-init` is on
    output_ids: Optional[List[int]]
    # Detokenization configs
    skip_special_tokens: List[bool]
    spaces_between_special_tokens: List[bool]
    no_stop_trim: List[bool]
    # Token counts
    prompt_tokens: List[int]
    completion_tokens: List[int]
    cached_tokens: List[int]
    # Logprobs
    input_token_logprobs_val: List[float]
    input_token_logprobs_idx: List[int]
    output_token_logprobs_val: List[float]
    output_token_logprobs_idx: List[int]
    input_top_logprobs_val: List[List]
    input_top_logprobs_idx: List[List]
    output_top_logprobs_val: List[List]
    output_top_logprobs_idx: List[List]
    normalized_prompt_logprob: List[float]


@dataclass
class BatchStrOut:
    # The request id
    rids: List[str]
    # The finish reason
    finished_reasons: List[dict]
    # The output decoded strings
    output_strs: List[str]

    # Token counts
    # real input and output tokens can be get from
    # origin_input_ids and output_ids by enabling --return_token_ids
    # TODO (Shuai): Rename this to clarify the meaning.
    prompt_tokens: List[int]
    completion_tokens: List[int]
    cached_tokens: List[int]

    # Logprobs
    input_token_logprobs_val: List[float]
    input_token_logprobs_idx: List[int]
    output_token_logprobs_val: List[float]
    output_token_logprobs_idx: List[int]
    input_top_logprobs_val: List[List]
    input_top_logprobs_idx: List[List]
    output_top_logprobs_val: List[List]
    output_top_logprobs_idx: List[List]
    normalized_prompt_logprob: List[float]


@dataclass
class BatchEmbeddingOut:
    # The request id
    rids: List[str]
    # The output embedding
    embeddings: List[List[float]]
    # The meta info
    meta_info: List[Dict]
    # The finish reason
    finished_reason: List[BaseFinishReason]


class ProfileReq(Enum):
    START_PROFILE = 1
    STOP_PROFILE = 2


@dataclass
class FlushCacheReq:
    pass


@dataclass
class UpdateWeightReqInput:
    # The model path with the new weights
    model_path: str
    # The format to load the weights
    load_format: Optional[str] = None


@dataclass
class UpdateWeightReqOutput:
    success: bool
    message: str


@dataclass
class AbortReq:
    # The request id
    rid: str


@dataclass
class TokenizedRewardReqInput:
    # The request id
    rid: str
    # The input text
    input_text: str
    # The input token ids
    input_ids: List[int]
    # Dummy sampling params for compatibility
    sampling_params: SamplingParams


@dataclass
class RewardReqInput:
    # The input prompt in the chat format. It can be a single prompt or a batch of prompts.
    conv: Union[List[List[Dict]], List[Dict]]
    # The request id.
    rid: Optional[Union[List[str], str]] = None
    # Dummy sampling params for compatibility
    sampling_params: Union[List[Dict], Dict] = None

    is_single: bool = True

    def post_init(self):
        self.is_single = isinstance(self.conv[0], dict)

        if self.is_single:
            if self.rid is None:
                self.rid = uuid.uuid4().hex
            if self.sampling_params is None:
                self.sampling_params = {}
            self.sampling_params["max_new_tokens"] = 1
        else:
            # support select operation
            self.batch_size = len(self.conv)
            if self.rid is None:
                self.rid = [uuid.uuid4().hex for _ in range(self.batch_size)]
            else:
                if not isinstance(self.rid, list):
                    raise ValueError("The rid should be a list.")
            if self.sampling_params is None:
                self.sampling_params = [{}] * self.batch_size
            for i in range(self.batch_size):
                self.sampling_params[i]["max_new_tokens"] = 1


@dataclass
class GetMemPoolSizeReq:
    pass


@dataclass
class GetMemPoolSizeReqOutput:
    size: int


@dataclass
class RegisterToppingsReqInput:
    model_path_or_name: str
    topping_type: str
    served_name: Optional[str] = None

    def post_init(self):
        if self.topping_type not in ["delta", "lora", "full"]:
            raise ValueError(f"The type should be either 'delta', 'lora' or 'full'")
        if not self.served_name:
            self.served_name = self.model_path_or_name


@dataclass
class UpdateWeightFromDiskReqOutput:
    success: bool
    message: str
