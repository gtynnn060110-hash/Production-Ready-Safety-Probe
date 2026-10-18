"""Pointwise linear baseline; temporal readout is deliberately separate."""

from torch import nn

from ..contracts import FeatureBatch, ProbeOutput


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_heads: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_heads)

    def forward(self, batch: FeatureBatch) -> ProbeOutput:
        return ProbeOutput(
            self.classifier(batch.hidden_states), batch.response_mask & batch.attention_mask
        )
