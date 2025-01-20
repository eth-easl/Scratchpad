import time
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field
from typing_extensions import Literal


class ModelCard(BaseModel):
    """Model cards."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "scratchpad"
    root: Optional[str] = None


class FunctionResponse(BaseModel):
    """Function response."""

    name: str
    arguments: str


class ToolCall(BaseModel):
    """Tool call response."""

    id: str
    type: Literal["function"] = "function"
    function: FunctionResponse


class Function(BaseModel):
    """Function descriptions."""

    description: Optional[str] = Field(default=None, examples=[None])
    name: str
    parameters: Optional[object] = None


class Tool(BaseModel):
    """Function wrapper."""

    type: str = Field(default="function", examples=["function"])
    function: Function


class ToolChoiceFuncName(BaseModel):
    """The name of tool choice function."""

    name: str


class ToolChoice(BaseModel):
    """The tool choice definition."""

    function: ToolChoiceFuncName
    type: Literal["function"] = Field(default="function", examples=["function"])


class ModelList(BaseModel):
    """Model list consists of model cards."""

    object: str = "list"
    data: List[ModelCard] = []


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: int


class LogProbs(BaseModel):
    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[Optional[float]] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: List[Optional[Dict[str, float]]] = Field(default_factory=list)


class TopLogprob(BaseModel):
    token: str
    bytes: List[int]
    logprob: float


class ChatCompletionTokenLogprob(BaseModel):
    token: str
    bytes: List[int]
    logprob: float
    top_logprobs: List[TopLogprob]


class ChoiceLogprobs(BaseModel):
    # build for v1/chat/completions response
    content: List[ChatCompletionTokenLogprob]


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class StreamOptions(BaseModel):
    include_usage: Optional[bool] = False


class JsonSchemaResponseFormat(BaseModel):
    name: str
    description: Optional[str] = None
    # use alias to workaround pydantic conflict
    schema_: Optional[Dict[str, object]] = Field(alias="schema", default=None)
    strict: Optional[bool] = False


class FileRequest(BaseModel):
    # https://platform.openai.com/docs/api-reference/files/create
    file: bytes  # The File object (not file name) to be uploaded
    purpose: str = (
        "batch"  # The intended purpose of the uploaded file, default is "batch"
    )


class FileResponse(BaseModel):
    id: str
    object: str = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str


class FileDeleteResponse(BaseModel):
    id: str
    object: str = "file"
    deleted: bool


class BatchRequest(BaseModel):
    input_file_id: (
        str  # The ID of an uploaded file that contains requests for the new batch
    )
    endpoint: str  # The endpoint to be used for all requests in the batch
    completion_window: str  # The time frame within which the batch should be processed
    metadata: Optional[dict] = None  # Optional custom metadata for the batch


class BatchResponse(BaseModel):
    id: str
    object: str = "batch"
    endpoint: str
    errors: Optional[dict] = None
    input_file_id: str
    completion_window: str
    status: str = "validating"
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None
    created_at: int
    in_progress_at: Optional[int] = None
    expires_at: Optional[int] = None
    finalizing_at: Optional[int] = None
    completed_at: Optional[int] = None
    failed_at: Optional[int] = None
    expired_at: Optional[int] = None
    cancelling_at: Optional[int] = None
    cancelled_at: Optional[int] = None
    request_counts: dict = {"total": 0, "completed": 0, "failed": 0}
    metadata: Optional[dict] = None


class CompletionRequest(BaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/completions/create
    model: str
    prompt: Union[List[int], List[List[int]], str, List[str]]
    best_of: Optional[int] = None
    echo: Optional[bool] = False
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[int] = None
    max_tokens: Optional[int] = 16
    n: int = 1
    presence_penalty: Optional[float] = 0.0
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    suffix: Optional[str] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    user: Optional[str] = None

    # Extra parameters for SRT backend only and will be ignored by OpenAI models.
    regex: Optional[str] = None
    json_schema: Optional[str] = None
    ignore_eos: Optional[bool] = False
    min_tokens: Optional[int] = 0
    repetition_penalty: Optional[float] = 1.0
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo


class CompletionResponseStreamChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[str] = None


class CompletionStreamResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = None


class ChatCompletionMessageContentTextPart(BaseModel):
    type: Literal["text"]
    text: str


class ChatCompletionMessageContentImageURL(BaseModel):
    url: str
    detail: Optional[Literal["auto", "low", "high"]] = "auto"


class ChatCompletionMessageContentImagePart(BaseModel):
    type: Literal["image_url"]
    image_url: ChatCompletionMessageContentImageURL
    modalities: Optional[Literal["image", "multi-images", "video"]] = "image"


ChatCompletionMessageContentPart = Union[
    ChatCompletionMessageContentTextPart, ChatCompletionMessageContentImagePart
]


class ChatCompletionMessageGenericParam(BaseModel):
    role: Literal["system", "assistant"]
    content: Union[str, List[ChatCompletionMessageContentTextPart]]


class ChatCompletionMessageUserParam(BaseModel):
    role: Literal["user"]
    content: Union[str, List[ChatCompletionMessageContentPart]]


ChatCompletionMessageParam = Union[
    ChatCompletionMessageGenericParam, ChatCompletionMessageUserParam
]


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object", "json_schema"]
    json_schema: Optional[JsonSchemaResponseFormat] = None


class ChatCompletionRequest(BaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/chat/create
    messages: List[ChatCompletionMessageParam]
    model: str
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = 1
    presence_penalty: Optional[float] = 0.0
    response_format: Optional[ResponseFormat] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    user: Optional[str] = None
    tools: Optional[List[Tool]] = Field(default=None, examples=[None])
    tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = Field(
        default="auto", examples=["none"]
    )
    # Extra parameters
    regex: Optional[str] = None
    min_tokens: Optional[int] = 0
    repetition_penalty: Optional[float] = 1.0
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    skip_special_tokens: bool = True
    session_params: Optional[Dict] = None
    regex: Optional[str] = None
    ebnf: Optional[str] = None
    top_k: Optional[int] = None
    ignore_eos: bool = False
    no_stop_trim: bool = False


class ChatMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    logprobs: Optional[Union[LogProbs, ChoiceLogprobs]] = None
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    logprobs: Optional[Union[LogProbs, ChoiceLogprobs]] = None
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = None


class EmbeddingRequest(BaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/embeddings/create
    input: Union[List[int], List[List[int]], str, List[str]]
    model: str
    encoding_format: str = "float"
    dimensions: int = None
    user: Optional[str] = None


class EmbeddingObject(BaseModel):
    embedding: List[float]
    index: int
    object: str = "embedding"


class EmbeddingResponse(BaseModel):
    data: List[EmbeddingObject]
    model: str
    object: str = "list"
    usage: Optional[UsageInfo] = None
