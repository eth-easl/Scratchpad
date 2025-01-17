import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, model_validator

from scratchpad.utils import logger


class OpenAIBaseModel(BaseModel):
    # OpenAI API does allow extra fields
    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def __log_extra_fields__(cls, data):
        if isinstance(data, dict):
            # Get all class field names and their potential aliases
            field_names = set()
            for field_name, field in cls.model_fields.items():
                field_names.add(field_name)
                if hasattr(field, "alias") and field.alias:
                    field_names.add(field.alias)

            # Compare against both field names and aliases
            extra_fields = data.keys() - field_names
            if extra_fields:
                logger.warning(
                    "The following fields were present in the request "
                    "but ignored: %s",
                    extra_fields,
                )
        return data


@dataclass
class GenerateReqInput:
    # The input prompt. It can be a single prompt or a batch of prompts.
    text: Optional[Union[List[str], str]] = None
    # The token ids for text; one can either specify text or input_ids.
    model: Optional[str] = None
    echo: bool = False
    input_ids: Optional[Union[List[List[int]], List[int]]] = None
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

    is_single: bool = True

    # LoRA related
    lora_path: Optional[Union[List[Optional[str]], Optional[str]]] = None

    def post_init(self):
        if (self.text is None and self.input_ids is None) or (
            self.text is not None and self.input_ids is not None
        ):
            raise ValueError("Either text or input_ids should be provided.")

        if (
            isinstance(self.sampling_params, dict)
            and self.sampling_params.get("n", 1) != 1
        ):
            is_single = False
        else:
            if self.text is not None:
                is_single = isinstance(self.text, str)
            else:
                is_single = isinstance(self.input_ids[0], int)
        self.is_single = is_single

        if is_single:
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
            parallel_sample_num_list = []
            if isinstance(self.sampling_params, dict):
                parallel_sample_num = self.sampling_params.get("n", 1)
            elif isinstance(self.sampling_params, list):
                for sp in self.sampling_params:
                    parallel_sample_num = sp.get("n", 1)
                    parallel_sample_num_list.append(parallel_sample_num)
                parallel_sample_num = max(parallel_sample_num_list)
                all_equal = all(
                    element == parallel_sample_num
                    for element in parallel_sample_num_list
                )
                if parallel_sample_num > 1 and (not all_equal):
                    # TODO cope with the case that the parallel_sample_num is different for different samples
                    raise ValueError(
                        "The parallel_sample_num should be the same for all samples in sample params."
                    )
            else:
                parallel_sample_num = 1
            self.parallel_sample_num = parallel_sample_num

            if parallel_sample_num != 1:
                # parallel sampling +1 represents the original prefill stage
                num = parallel_sample_num + 1
                if isinstance(self.text, list):
                    # suppot batch operation
                    self.batch_size = len(self.text)
                    num = num * len(self.text)
                elif isinstance(self.input_ids, list) and isinstance(
                    self.input_ids[0], list
                ):
                    self.batch_size = len(self.input_ids)
                    num = num * len(self.input_ids)
                else:
                    self.batch_size = 1
            else:
                # support select operation
                num = len(self.text) if self.text is not None else len(self.input_ids)
                self.batch_size = num

            if self.image_data is None:
                self.image_data = [None] * num

            elif not isinstance(self.image_data, list):
                self.image_data = [self.image_data] * num
            elif isinstance(self.image_data, list):
                # multi-image with n > 1
                self.image_data = self.image_data * num

            if self.sampling_params is None:
                self.sampling_params = [{}] * num
            elif not isinstance(self.sampling_params, list):
                self.sampling_params = [self.sampling_params] * num

            if self.rid is None:
                self.rid = [uuid.uuid4().hex for _ in range(num)]
            else:
                if not isinstance(self.rid, list):
                    raise ValueError("The rid should be a list.")

            if self.return_logprob is None:
                self.return_logprob = [False] * num
            elif not isinstance(self.return_logprob, list):
                self.return_logprob = [self.return_logprob] * num

            if self.logprob_start_len is None:
                self.logprob_start_len = [-1] * num
            elif not isinstance(self.logprob_start_len, list):
                self.logprob_start_len = [self.logprob_start_len] * num

            if self.top_logprobs_num is None:
                self.top_logprobs_num = [0] * num
            elif not isinstance(self.top_logprobs_num, list):
                self.top_logprobs_num = [self.top_logprobs_num] * num


class ErrorResponse(OpenAIBaseModel):
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: int
