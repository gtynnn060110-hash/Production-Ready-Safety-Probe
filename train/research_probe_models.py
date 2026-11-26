"""Image-pipeline adapters for the text_eval MQA-RMAttn and SingProbe models.

Both probes consume the same full multimodal hidden-state sequence produced by
the image pipeline. Their logits remain token-level; sequence readout is done
by max over valid positions during validation/evaluation.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalMQA(nn.Module):
    """Causal multi-query attention encoder, matching text_eval's adapter."""

    def __init__(
        self,
        input_dim: int,
        num_query_heads: int = 4,
        num_kv_heads: int = 1,
        head_dim: int = 64,
        sliding_window: int = 0,
        rms_norm_eps: float = 1e-6,
        attention_chunk_size: int = 256,
    ) -> None:
        super().__init__()
        if num_kv_heads != 1:
            raise ValueError("These experiments use one shared KV head (MQA).")
        if min(input_dim, num_query_heads, head_dim, attention_chunk_size) < 1:
            raise ValueError("Attention dimensions and chunk size must be positive.")
        if sliding_window < 0 or rms_norm_eps <= 0:
            raise ValueError("Invalid attention window or RMSNorm epsilon.")
        self.heads = num_query_heads
        self.dim = head_dim
        self.window = sliding_window
        self.chunk = attention_chunk_size
        width = num_query_heads * head_dim
        self.proj_q = nn.Linear(input_dim, width, bias=False)
        self.proj_k = nn.Linear(input_dim, head_dim, bias=False)
        self.proj_v = nn.Linear(input_dim, head_dim, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)
        self.norm = nn.RMSNorm(width, eps=rms_norm_eps)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch, length, _ = hidden.shape
        query = self.proj_q(hidden)
        q = query.view(batch, length, self.heads, self.dim).transpose(1, 2)
        k = self.proj_k(hidden).unsqueeze(1)
        v = self.proj_v(hidden).unsqueeze(1)
        valid = attention_mask.to(device=hidden.device, dtype=torch.bool)
        context = []
        for start in range(0, length, self.chunk):
            end = min(start + self.chunk, length)
            left = max(0, start - self.window + 1) if self.window else 0
            qi = torch.arange(start, end, device=hidden.device)
            ki = torch.arange(left, end, device=hidden.device)
            allowed = ki[None, :] <= qi[:, None]
            if self.window:
                allowed &= ki[None, :] > qi[:, None] - self.window
            mask = allowed[None, None] & valid[:, None, None, left:end]
            context.append(
                F.scaled_dot_product_attention(
                    q[:, :, start:end],
                    k[:, :, left:end],
                    v[:, :, left:end],
                    attn_mask=mask,
                    enable_gqa=self.heads != 1,
                )
            )
        mixed = torch.cat(context, dim=2).transpose(1, 2).reshape(batch, length, -1)
        return self.norm(query + self.o_proj(mixed))


class MQARMAttnProbe(nn.Module):
    """Full-causal MQA context encoder + rolling multi-head attention readout."""

    def __init__(
        self,
        input_dim: int,
        num_tasks: int = 4,
        num_query_heads: int = 4,
        num_kv_heads: int = 1,
        head_dim: int = 64,
        context_window: int = 0,
        risk_window: int = 40,
        rms_norm_eps: float = 1e-6,
        attention_chunk_size: int = 256,
        mlp_hidden_sizes: tuple[int, int] = (100, 100),
        internal_heads: int = 10,
        context_ffn_hidden_size: int = 0,
    ) -> None:
        super().__init__()
        if len(mlp_hidden_sizes) != 2 or min(*mlp_hidden_sizes, internal_heads, risk_window) < 1:
            raise ValueError("Invalid RMAttn MLP/head/window configuration.")
        self.hidden_size = input_dim
        self.num_tasks = num_tasks
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.context_window = context_window
        self.risk_window = risk_window
        self.rms_norm_eps = rms_norm_eps
        self.attention_chunk_size = attention_chunk_size
        self.mlp_hidden_sizes = tuple(mlp_hidden_sizes)
        self.internal_heads = internal_heads
        self.context = CausalMQA(
            input_dim, num_query_heads, num_kv_heads, head_dim,
            context_window, rms_norm_eps, attention_chunk_size,
        )
        width = num_query_heads * head_dim
        first, second = mlp_hidden_sizes
        if type(context_ffn_hidden_size) is not int or context_ffn_hidden_size < 0:
            raise ValueError("context_ffn_hidden_size must be a non-negative integer")
        self.context_ffn_hidden_size = context_ffn_hidden_size
        if context_ffn_hidden_size:
            self.context_ffn = nn.Sequential(nn.Linear(width, context_ffn_hidden_size), nn.GELU(), nn.Linear(context_ffn_hidden_size, width))
            self.context_ffn_norm = nn.RMSNorm(width, eps=rms_norm_eps)
        else:
            self.context_ffn = None
        self.mlp = nn.Sequential(nn.Linear(width, first), nn.ReLU(), nn.Linear(first, second))
        self.query = nn.Linear(second, internal_heads * num_tasks, bias=False)
        self.value = nn.Linear(second, internal_heads * num_tasks, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        valid = attention_mask.to(device=hidden.device, dtype=torch.bool)
        contextual = self.context(hidden, valid)
        if self.context_ffn is not None:
            contextual = self.context_ffn_norm(contextual + self.context_ffn(contextual))
        y = self.mlp(contextual)
        batch, length, _ = y.shape
        scores = self.query(y).view(batch, length, self.internal_heads, self.num_tasks).float()
        values = self.value(y).view(batch, length, self.internal_heads, self.num_tasks).float()
        width = self.risk_window
        scores = F.pad(scores, (0, 0, 0, 0, width - 1, 0)).unfold(1, width, 1)
        values = F.pad(values, (0, 0, 0, 0, width - 1, 0)).unfold(1, width, 1)
        window_mask = F.pad(valid, (width - 1, 0), value=False).unfold(1, width, 1)
        window_mask = window_mask & valid.unsqueeze(-1)
        window_mask = window_mask[:, :, None, None, :]
        weights = scores.masked_fill(~window_mask, -1e30).softmax(-1) * window_mask
        logits = (weights * values).sum(-1).sum(2) + self.bias.float()
        return logits.masked_fill(~valid.unsqueeze(-1), 0)


class SingProbe(nn.Module):
    """Full-causal MQA context encoder + per-token linear task classifier."""

    def __init__(
        self,
        input_dim: int,
        num_tasks: int = 4,
        num_query_heads: int = 4,
        num_kv_heads: int = 1,
        head_dim: int = 64,
        sliding_window: int = 0,
        rms_norm_eps: float = 1e-6,
        init_bias: float = 0.0,
        attention_chunk_size: int = 256,
    ) -> None:
        super().__init__()
        self.hidden_size = input_dim
        self.num_tasks = num_tasks
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.rms_norm_eps = rms_norm_eps
        self.init_bias = init_bias
        self.attention_chunk_size = attention_chunk_size
        self.context = CausalMQA(
            input_dim, num_query_heads, num_kv_heads, head_dim,
            sliding_window, rms_norm_eps, attention_chunk_size,
        )
        self.classifier = nn.Linear(num_query_heads * head_dim, num_tasks)
        nn.init.constant_(self.classifier.bias, init_bias)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        valid = attention_mask.to(device=hidden.device, dtype=torch.bool)
        logits = self.classifier(self.context(hidden, valid))
        return logits.masked_fill(~valid.unsqueeze(-1), 0)


def singprobe_weighted_loss(
    token_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    targets: torch.Tensor,
    *,
    temperature: float = 0.1,
    beta_tau: float = 0.35,
    detach_weights: bool = True,
    label_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Text-eval SingProbeWeighted objective, applied to full image sequences."""
    valid = attention_mask.to(device=token_logits.device, dtype=torch.bool)
    z = token_logits.float()
    if label_mask is None:
        label_mask = torch.ones(
            (z.shape[0], z.shape[2]), dtype=torch.bool, device=z.device
        )
    else:
        label_mask = label_mask.to(device=z.device, dtype=torch.bool)
    mask = valid[:, :, None] & label_mask[:, None, :]
    valid_examples = label_mask & valid.any(1, keepdim=True)
    z = torch.where(mask, z, 0.0)
    probability = torch.where(mask, z.sigmoid(), 0.0)
    if detach_weights:
        probability = probability.detach()
    count = mask.sum((0, 1)).clamp_min(1).float()
    mean = (probability * mask).sum((0, 1)) / count
    variance = ((probability - mean).square() * mask).sum((0, 1)) / count
    std = torch.where(variance > 0, variance.clamp_min(1e-20).sqrt(), variance * 0)
    beta = (std / beta_tau).clamp(0, 1)
    scores = probability * beta / temperature
    weights = scores.masked_fill(~mask, -1e30).softmax(dim=1) * mask
    labels = targets.to(device=z.device, dtype=z.dtype)
    labels = torch.where(label_mask, labels, 0.0)[:, None, :].expand_as(z)
    bce = F.binary_cross_entropy_with_logits(
        z, labels, reduction="none"
    )
    per_example_head = (weights * bce).sum(dim=1)
    head_counts = valid_examples.sum(0)
    per_head = (per_example_head * valid_examples).sum(0) / head_counts.clamp_min(1)
    active_heads = head_counts > 0
    return (per_head * active_heads).sum() / active_heads.sum().clamp_min(1)


def max_readout(token_logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    valid = attention_mask.to(device=token_logits.device, dtype=torch.bool)
    if not valid.any(dim=1).all():
        raise ValueError("Cannot take a max readout over an empty sequence.")
    return token_logits.float().masked_fill(~valid.unsqueeze(-1), -torch.inf).amax(dim=1)
