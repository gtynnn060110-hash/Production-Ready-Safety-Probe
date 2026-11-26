"""Gemini-inspired RMAttn; fixed learned queries, not token self-attention.

Token-loss adaptation: sum internal heads BEFORE temporal readout. Trailing
partial-prefix windows produce a local score at every response position.
"""

import torch
from torch import nn
from torch.nn import functional as F

from ..contracts import FeatureBatch, ProbeOutput


class GeminiProbe(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        mlp_hidden_sizes=(100, 100),
        internal_heads: int = 10,
        window_size: int = 10,
        partial_prefix_windows: bool = True,
    ):
        super().__init__()
        if not partial_prefix_windows:
            raise ValueError("Token-loss experiments require partial-prefix windows")
        if len(mlp_hidden_sizes) != 2 or min(*mlp_hidden_sizes, internal_heads, window_size) < 1:
            raise ValueError("Invalid Gemini MLP/head/window configuration")
        self.heads, self.classes, self.window = internal_heads, num_heads, window_size
        a, b = mlp_hidden_sizes
        self.mlp = nn.Sequential(nn.Linear(input_dim, a), nn.ReLU(), nn.Linear(a, b))
        self.query = nn.Linear(b, internal_heads * num_heads, bias=False)
        self.value = nn.Linear(b, internal_heads * num_heads, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_heads))

    def forward(self, batch: FeatureBatch) -> ProbeOutput:
        mask = batch.response_mask & batch.attention_mask
        y = self.mlp(batch.hidden_states)
        b, t, _ = y.shape
        scores = self.query(y).view(b, t, self.heads, self.classes).float()
        values = self.value(y).view(b, t, self.heads, self.classes).float()
        w = self.window
        scores = F.pad(scores, (0, 0, 0, 0, w - 1, 0)).unfold(1, w, 1)
        values = F.pad(values, (0, 0, 0, 0, w - 1, 0)).unfold(1, w, 1)
        # A window must not bridge prompt/padding or disjoint response spans.
        starts = mask & ~F.pad(mask[:, :-1], (1, 0), value=False)
        spans = starts.long().cumsum(1)
        window_spans = F.pad(spans, (w - 1, 0), value=-1).unfold(1, w, 1)
        valid = F.pad(mask, (w - 1, 0), value=False).unfold(1, w, 1)
        valid = valid & (window_spans == spans.unsqueeze(-1)) & mask.unsqueeze(-1)
        valid = valid[:, :, None, None, :]
        weights = scores.masked_fill(~valid, -1e30).softmax(-1) * valid
        components = (weights * values).sum(-1)
        logits = components.sum(2) + self.bias.float()
        return ProbeOutput(logits, mask)
