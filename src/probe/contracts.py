"""Shared data contracts. Importing the framework does not import torch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from torch import Tensor

HEADS = ("privacy", "misinformation", "harmful_language", "malicious_uses", "overall")


@dataclass
class Sample:
    id: str
    prompt: str
    response: str
    labels: list[float]  # [C], in HEADS order; 1 means unsafe
    label_mask: list[bool]  # [C], unknown labels are masked, never treated as negatives


@dataclass
class FeatureBatch:
    ids: list[str]
    hidden_states: Tensor  # [B, T, D], selected layer features concatenated along D
    attention_mask: Tensor  # [B, T], real context, including prompt
    response_mask: Tensor  # [B, T], only original response content
    labels: Tensor  # [B, C]
    label_mask: Tensor  # [B, C]


@dataclass
class ProbeOutput:
    local_logits: Tensor  # [B, T, C], no time pooling or cumulative max
    local_mask: Tensor  # [B, T], supervised response positions


class Probe(Protocol):
    def __call__(self, batch: FeatureBatch) -> ProbeOutput: ...


class Loss(Protocol):
    def __call__(self, output: ProbeOutput, labels: Tensor, label_mask: Tensor) -> Tensor: ...
