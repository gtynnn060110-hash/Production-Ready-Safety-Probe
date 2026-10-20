"""SingProbe-style causal MQA, output projection, query residual, and RMSNorm.

Adapted from inclusionAI/SingProbe (10edcb8), models/sglang_attn.py.
Padding-aware blocked SDPA avoids a full T-by-T mask for long responses.
"""

import torch
from torch import nn
from torch.nn import functional as F

from ..contracts import FeatureBatch, ProbeOutput


class CausalMQA(nn.Module):
    """Context encoder shared by the standalone and stacked probes."""

    def __init__(
        self,
        input_dim: int,
        num_query_heads: int = 4,
        num_kv_heads: int = 1,
        head_dim: int = 64,
        sliding_window: int = 2048,
        rms_norm_eps: float = 1e-6,
        attention_chunk_size: int = 256,
    ):
        super().__init__()
        if num_kv_heads != 1:
            raise ValueError("SingProbe uses exactly one shared KV head (MQA)")
        if min(num_query_heads, head_dim, attention_chunk_size) < 1 or sliding_window < 0:
            raise ValueError("Invalid SingProbe head/window/chunk configuration")
        if rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")
        self.heads, self.dim = num_query_heads, head_dim
        self.window, self.chunk = sliding_window, attention_chunk_size
        width = num_query_heads * head_dim
        self.proj_q = nn.Linear(input_dim, width, bias=False)
        self.proj_k = nn.Linear(input_dim, head_dim, bias=False)
        self.proj_v = nn.Linear(input_dim, head_dim, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)
        self.norm = nn.RMSNorm(width, eps=rms_norm_eps)

    def forward(self, batch: FeatureBatch) -> torch.Tensor:
        x = batch.hidden_states
        b, t, _ = x.shape
        query = self.proj_q(x)
        q = query.view(b, t, self.heads, self.dim).transpose(1, 2)
        k = self.proj_k(x).unsqueeze(1)
        v = self.proj_v(x).unsqueeze(1)
        context = []
        for start in range(0, t, self.chunk):
            end = min(start + self.chunk, t)
            left = max(0, start - self.window + 1) if self.window else 0
            qi = torch.arange(start, end, device=x.device)
            ki = torch.arange(left, end, device=x.device)
            allowed = ki[None, :] <= qi[:, None]
            if self.window:
                allowed = allowed & (ki[None, :] > qi[:, None] - self.window)
            mask = allowed[None, None] & batch.attention_mask[:, None, None, left:end]
            context.append(
                F.scaled_dot_product_attention(
                    q[:, :, start:end],
                    k[:, :, left:end],
                    v[:, :, left:end],
                    attn_mask=mask,
                    enable_gqa=self.heads != 1,
                )
            )
        mixed = torch.cat(context, dim=2).transpose(1, 2).reshape(b, t, -1)
        return self.norm(query + self.o_proj(mixed))


class SingProbe(CausalMQA):
    """Standalone encoder and classifier; original state-dict keys are preserved."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        num_query_heads: int = 4,
        num_kv_heads: int = 1,
        head_dim: int = 64,
        sliding_window: int = 2048,
        rms_norm_eps: float = 1e-6,
        init_bias: float = 0.0,
        attention_chunk_size: int = 256,
    ):
        super().__init__(
            input_dim=input_dim,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            sliding_window=sliding_window,
            rms_norm_eps=rms_norm_eps,
            attention_chunk_size=attention_chunk_size,
        )
        self.classifier = nn.Linear(num_query_heads * head_dim, num_heads)
        nn.init.constant_(self.classifier.bias, init_bias)

    def forward(self, batch: FeatureBatch) -> ProbeOutput:
        logits = self.classifier(super().forward(batch))
        return ProbeOutput(logits, batch.response_mask & batch.attention_mask)
