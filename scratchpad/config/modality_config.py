from dataclasses import dataclass, field
from typing import Mapping


@dataclass
class MultiModalConfig:
    """Controls the behavior of multimodal models."""

    limit_per_prompt: Mapping[str, int] = field(default_factory=dict)
    """
    The maximum number of multi-modal input instances allowed per prompt
    for each :class:`~vllm.multimodal.MultiModalPlugin`.
    """

    # TODO: Add configs to init vision tower or not.
