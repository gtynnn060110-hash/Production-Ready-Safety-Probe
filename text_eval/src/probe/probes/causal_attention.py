"""Two-layer MLP with task-specific causal attention pooling."""

import torch
from torch import nn

from ..contracts import FeatureBatch, ProbeOutput


class CausalAttentionProbe(nn.Module):
    """Emit one local logit per task and token, attending only to causal context."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        mlp_hidden_size: int = 256,
        attention_heads: int = 8,
        dropout: float = 0.1,
        query_chunk_size: int = 128,
    ):
        super().__init__()
        if min(input_dim, num_heads, mlp_hidden_size, attention_heads, query_chunk_size) < 1:
            raise ValueError("dimensions, head counts, and query_chunk_size must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.input_dim = input_dim
        self.mlp_hidden_size = mlp_hidden_size
        self.attention_heads = attention_heads
        self.num_heads = num_heads
        self.query_chunk_size = query_chunk_size
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, mlp_hidden_size),
            nn.ReLU(),
        )
        self.queries = nn.Parameter(torch.empty(num_heads, attention_heads, mlp_hidden_size))
        self.values = nn.Parameter(torch.empty(num_heads, attention_heads, mlp_hidden_size))
        nn.init.normal_(self.queries, std=mlp_hidden_size**-0.5)
        nn.init.normal_(self.values, std=mlp_hidden_size**-0.5)

    def forward(self, batch: FeatureBatch) -> ProbeOutput:
        hidden = batch.hidden_states
        if hidden.ndim != 3 or hidden.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected [batch, tokens, {self.input_dim}] features, got {tuple(hidden.shape)}"
            )
        batch_size, sequence_length, _ = hidden.shape
        valid = batch.attention_mask.bool()
        reps = self.mlp(hidden)
        scores = torch.einsum("btd,khd->bkht", reps, self.queries).float()
        values = torch.einsum("btd,khd->bkht", reps, self.values).float()
        scale = self.mlp_hidden_size**-0.5
        key_positions = torch.arange(sequence_length, device=hidden.device)
        outputs = []
        for start in range(0, sequence_length, self.query_chunk_size):
            end = min(start + self.query_chunk_size, sequence_length)
            query_positions = torch.arange(start, end, device=hidden.device)
            attention_scores = scores[:, :, :, None, :].expand(
                -1, -1, -1, end - start, -1
            ) * scale
            allowed = valid[:, None, None, None, :] & (
                key_positions[None, None, None, None, :]
                <= query_positions[None, None, None, :, None]
            )
            attention_scores = attention_scores.masked_fill(
                ~allowed, torch.finfo(attention_scores.dtype).min
            )
            query_valid = valid[:, None, None, start:end, None]
            attention_scores = torch.where(
                query_valid, attention_scores, torch.zeros_like(attention_scores)
            )
            weights = attention_scores.softmax(dim=-1)
            pooled = (weights * values[:, :, :, None, :]).sum(-1).sum(2)
            outputs.append(pooled.transpose(1, 2))
        logits = torch.cat(outputs, dim=1).to(hidden.dtype)
        logits = logits.masked_fill(~valid.unsqueeze(-1), 0)
        return ProbeOutput(logits, batch.response_mask.bool() & valid)
