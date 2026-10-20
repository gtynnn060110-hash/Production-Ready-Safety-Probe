"""Probe architectures for token-level hidden-state sequences."""

from __future__ import annotations

import torch
from torch import nn

HEAD_NAMES = ("physical_harm", "hate_speech", "privacy", "overall_safety")
HEAD_CATEGORIES = {
    "physical_harm": {"violence_and_physical_harm"},
    "hate_speech": {"discriminatory_depictions", "insulting_and_harassing_behavior"},
    "privacy": {"privacy_invasion_and_surveillance"},
}


def targets_for_row(row: dict) -> torch.Tensor:
    """One-vs-rest risk labels plus overall response safety.

    BeaverTails-V supplies one topic category per example. The matching risk
    head receives the response-safety label; other risk heads receive explicit
    negatives, teaching category specificity. Overall safety uses every row.
    """
    label = row.get("label")
    if label not in (0, 1):
        raise ValueError(f"Expected binary response-safety label, got {label!r}")
    category = row.get("category")
    targets = [0.0, 0.0, 0.0, float(label)]
    if int(label) == 1:
        for index, name in enumerate(HEAD_NAMES[:3]):
            if category in HEAD_CATEGORIES[name]:
                targets[index] = 1.0
    return torch.tensor(targets, dtype=torch.float32)


class MaxOfRollingMeansProbe(nn.Module):
    """Multi-head linear token scores followed by max rolling-mean pooling."""

    def __init__(self, hidden_size: int, window_size: int = 40, num_heads: int = 1) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if window_size < 1:
            raise ValueError("window_size must be positive")
        if num_heads < 1:
            raise ValueError("num_heads must be positive")
        self.token_scorer = nn.Linear(hidden_size, num_heads)
        self.window_size = window_size
        self.num_heads = num_heads

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return one max rolling-mean logit per sequence.

        Args:
            hidden: [batch, sequence, hidden_size] token states.
            attention_mask: optional [batch, sequence] 1=valid token, 0=padding.
        """
        if hidden.ndim != 3:
            raise ValueError(f"Expected [batch, sequence, hidden], got {tuple(hidden.shape)}")
        scores = self.token_scorer(hidden)
        if attention_mask is None:
            attention_mask = torch.ones(scores.shape[:2], dtype=torch.bool, device=scores.device)
        else:
            if attention_mask.shape != scores.shape[:2]:
                raise ValueError("attention_mask must match the batch and sequence dimensions")
            attention_mask = attention_mask.to(device=scores.device, dtype=torch.bool)

        width = min(self.window_size, scores.shape[1])
        mask = attention_mask.to(dtype=scores.dtype)
        window_scores = scores.masked_fill(~attention_mask.unsqueeze(-1), 0).unfold(1, width, 1)
        window_mask = mask.unfold(1, width, 1).unsqueeze(2)
        counts = window_mask.sum(dim=-1).clamp_min(1)
        means = (window_scores * window_mask).sum(dim=-1) / counts
        means = means.masked_fill(window_mask.sum(dim=-1) == 0, torch.finfo(means.dtype).min)
        logits = means.max(dim=1).values
        return logits.squeeze(-1) if self.num_heads == 1 else logits


class CausalAttentionProbe(nn.Module):
    """Shared token MLP followed by per-task, multi-subhead causal attention.

    Returns one logit per task and sequence position. Every position pools only
    over valid positions at or before itself, so the output can be consumed by
    token-level SWiM training and streaming score smoothing.
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_hidden_size: int = 256,
        attention_heads: int = 8,
        num_tasks: int = 4,
        dropout: float = 0.1,
        query_chunk_size: int = 128,
    ) -> None:
        super().__init__()
        if min(hidden_size, mlp_hidden_size, attention_heads, num_tasks, query_chunk_size) < 1:
            raise ValueError("hidden size, MLP width, attention heads, tasks, and chunk size must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.hidden_size = hidden_size
        self.mlp_hidden_size = mlp_hidden_size
        self.attention_heads = attention_heads
        self.num_tasks = num_tasks
        self.query_chunk_size = query_chunk_size

        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, mlp_hidden_size),
            nn.ReLU(),
        )
        # Each task owns its attention queries and scalar value projections.
        self.queries = nn.Parameter(torch.empty(num_tasks, attention_heads, mlp_hidden_size))
        self.values = nn.Parameter(torch.empty(num_tasks, attention_heads, mlp_hidden_size))
        nn.init.normal_(self.queries, std=mlp_hidden_size ** -0.5)
        nn.init.normal_(self.values, std=mlp_hidden_size ** -0.5)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return causal per-token logits with shape ``[batch, seq, tasks]``."""
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected [batch, sequence, {self.hidden_size}] hidden states, got {tuple(hidden.shape)}"
            )
        batch_size, sequence_length, _ = hidden.shape
        if attention_mask is None:
            valid = torch.ones((batch_size, sequence_length), dtype=torch.bool, device=hidden.device)
        else:
            if attention_mask.shape != (batch_size, sequence_length):
                raise ValueError("attention_mask must match the batch and sequence dimensions")
            valid = attention_mask.to(device=hidden.device, dtype=torch.bool)

        representations = self.mlp(hidden)
        # A learned query gives each token's attention compatibility score.
        # Each prefix then normalizes those scores over its causal history.
        attention_scores = torch.einsum("btd,khd->bkht", representations, self.queries)
        # [B, task, attention-head, key-position]
        value_scores = torch.einsum("btd,khd->bkht", representations, self.values)
        scale = self.mlp_hidden_size ** -0.5
        key_valid = valid[:, None, None, None, :]
        outputs = []
        key_positions = torch.arange(sequence_length, device=hidden.device)
        for start in range(0, sequence_length, self.query_chunk_size):
            end = min(start + self.query_chunk_size, sequence_length)
            query_positions = torch.arange(start, end, device=hidden.device)
            # [B, task, head, query-position, key-position]. The token score
            # is shared across prefixes; the causal mask changes which keys
            # each prefix can attend to.
            scores = attention_scores[:, :, :, None, :].expand(-1, -1, -1, end - start, -1) * scale
            allowed = key_valid & (key_positions[None, None, None, None, :] <= query_positions[None, None, None, :, None])
            scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
            # Avoid NaNs on left-padding query positions, which are discarded below.
            query_valid = valid[:, None, None, start:end, None]
            scores = torch.where(query_valid, scores, torch.zeros_like(scores))
            attention = torch.softmax(scores, dim=-1)
            pooled = (attention * value_scores[:, :, :, None, :]).sum(dim=-1).sum(dim=2)
            outputs.append(pooled.transpose(1, 2))  # [B, query-position, task]
        logits = torch.cat(outputs, dim=1)
        return logits.masked_fill(~valid.unsqueeze(-1), 0)


def swim_logits(
    token_logits: torch.Tensor,
    attention_mask: torch.Tensor | None,
    window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply causal sliding-window logit means and return their valid mask.

    Full windows are used whenever the sequence is at least ``window_size``
    long. Short sequences produce one mean over all available valid positions.
    """
    if token_logits.ndim != 3:
        raise ValueError(f"Expected [batch, sequence, tasks], got {tuple(token_logits.shape)}")
    if window_size < 1:
        raise ValueError("window_size must be positive")
    batch_size, sequence_length, num_tasks = token_logits.shape
    if attention_mask is None:
        valid = torch.ones((batch_size, sequence_length), dtype=torch.bool, device=token_logits.device)
    else:
        if attention_mask.shape != (batch_size, sequence_length):
            raise ValueError("attention_mask must match the batch and sequence dimensions")
        valid = attention_mask.to(device=token_logits.device, dtype=torch.bool)

    width = min(window_size, sequence_length)
    masked = token_logits * valid.unsqueeze(-1).to(token_logits.dtype)
    windows = masked.unfold(1, width, 1).transpose(-1, -2)
    window_valid = valid.unfold(1, width, 1)
    counts = window_valid.sum(dim=-1)
    means = windows.sum(dim=-2) / counts.clamp_min(1).unsqueeze(-1)
    # A valid training sequence normally has no padding. The mask handling also
    # makes this utility safe for padded batches and left-padded model inputs.
    eligible = counts > 0
    if sequence_length >= window_size:
        eligible = counts == window_size
    means = means.masked_fill(~eligible.unsqueeze(-1), 0)
    return means, eligible


def ema_max_probabilities(
    token_logits: torch.Tensor,
    attention_mask: torch.Tensor | None,
    alpha: float,
) -> torch.Tensor:
    """Return per-sequence max EMA of token probabilities, one score per task."""
    if not 0 < alpha <= 1:
        raise ValueError("EMA alpha must be in (0, 1]")
    if token_logits.ndim != 3:
        raise ValueError(f"Expected [batch, sequence, tasks], got {tuple(token_logits.shape)}")
    batch_size, sequence_length, _ = token_logits.shape
    if attention_mask is None:
        valid = torch.ones((batch_size, sequence_length), dtype=torch.bool, device=token_logits.device)
    else:
        valid = attention_mask.to(device=token_logits.device, dtype=torch.bool)
    probabilities = torch.sigmoid(token_logits)
    ema = torch.zeros_like(probabilities[:, 0])
    maxima = torch.zeros_like(ema)
    seen = torch.zeros((batch_size, 1), dtype=torch.bool, device=token_logits.device)
    for position in range(sequence_length):
        current_valid = valid[:, position:position + 1]
        updated = alpha * probabilities[:, position] + (1 - alpha) * ema
        # Standard EMA initialization uses the first observation rather than
        # biasing early scores toward zero.
        updated = torch.where(seen, updated, probabilities[:, position])
        ema = torch.where(current_valid, updated, ema)
        maxima = torch.where(current_valid, torch.maximum(maxima, ema), maxima)
        seen |= current_valid
    return torch.where(seen, maxima, torch.zeros_like(maxima))
