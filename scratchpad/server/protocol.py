from typing import Optional
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


class ErrorResponse(OpenAIBaseModel):
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: int
