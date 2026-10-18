"""FP32 response-only objectives; equal sample weight within each valid head."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..contracts import ProbeOutput


def head_mean(values, valid):
    counts = valid.sum(0)
    means = (values * valid).sum(0) / counts.clamp_min(1)
    return (means * (counts > 0)).sum() / (counts > 0).sum().clamp_min(1)


def inputs(output, labels, label_mask):
    z = output.local_logits.float()
    mask = output.local_mask.bool()
    valid = label_mask.bool() & mask.any(1, keepdim=True)
    # Unknown labels may be NaN in future adapters: sanitize before BCE.
    y = torch.where(label_mask, labels.float(), 0.0)
    z = torch.where(mask.unsqueeze(-1) & label_mask[:, None, :], z, 0.0)
    bce = F.binary_cross_entropy_with_logits(z, y[:, None].expand_as(z), reduction="none")
    return z, mask, valid, bce


class TokenBCE(nn.Module):
    def forward(self, output: ProbeOutput, labels, label_mask):
        _, mask, valid, bce = inputs(output, labels, label_mask)
        per_sample = (bce * mask.unsqueeze(-1)).sum(1) / mask.sum(1).clamp_min(1)[:, None]
        return head_mean(per_sample, valid)


class SingProbeWeighted(nn.Module):
    def __init__(
        self, temperature: float = 1.0, beta_tau: float = 0.35, detach_weights: bool = True
    ):
        super().__init__()
        if any(not math.isfinite(v) or v <= 0 for v in (temperature, beta_tau)):
            raise ValueError("temperature and beta_tau must be finite and positive")
        if type(detach_weights) is not bool:
            raise ValueError("detach_weights must be a boolean")
        self.temperature, self.tau, self.detach = temperature, beta_tau, detach_weights
        self.diagnostics = {}

    def confidence_weights(self, output, label_mask):
        mask = output.local_mask[:, :, None] & label_mask[:, None, :]
        p = torch.where(mask, output.local_logits.float(), 0.0).sigmoid()
        if self.detach:
            p = p.detach()
        count = mask.sum((0, 1))
        mean = (p * mask).sum((0, 1)) / count.clamp_min(1)
        var = ((p - mean).square() * mask).sum((0, 1)) / count.clamp_min(1)
        # Safe sqrt gradient at exactly zero (constant/single-token batches).
        std = torch.where(var > 0, var.clamp_min(1e-20).sqrt(), var * 0)
        beta = (std / self.tau).clamp(0, 1)
        scores = p * beta / self.temperature
        weights = scores.masked_fill(~mask, -1e30).softmax(dim=1) * mask
        self.diagnostics = {
            "beta": beta.detach().cpu().tolist(),
            "mean_ess": float(
                (1 / weights.square().sum(1).clamp_min(1e-20))[
                    label_mask & output.local_mask.any(1, keepdim=True)
                ]
                .mean()
                .detach()
            )
            if mask.any()
            else 0.0,
        }
        return weights

    def forward(self, output: ProbeOutput, labels, label_mask):
        _, _, valid, bce = inputs(output, labels, label_mask)
        weights = self.confidence_weights(output, label_mask)
        return head_mean((weights * bce).sum(1), valid)


class PlugGuardATC(nn.Module):
    def __init__(self, anchor_tokens: int = 10, lambda_tv: float = 0.01, lambda_mono: float = 0.01):
        super().__init__()
        if type(anchor_tokens) is not int or anchor_tokens < 1:
            raise ValueError("anchor_tokens must be a positive integer")
        if any(not math.isfinite(v) or v < 0 for v in (lambda_tv, lambda_mono)):
            raise ValueError("ATC regularization coefficients must be non-negative")
        self.anchors, self.tv, self.mono = anchor_tokens, lambda_tv, lambda_mono

    def forward(self, output: ProbeOutput, labels, label_mask):
        z, mask, valid, bce = inputs(output, labels, label_mask)
        length = mask.sum(1)
        n = torch.minimum(length // 2, torch.full_like(length, self.anchors))
        rank = mask.long().cumsum(1)
        first = mask & (rank <= n[:, None])
        last = mask & (rank > (length - n.clamp_min(1))[:, None])
        safe_bce = F.softplus(z)
        head = (safe_bce * first[:, :, None]).sum(1) / n.clamp_min(1)[:, None]
        tail = (bce * last[:, :, None]).sum(1) / n.clamp_min(1)[:, None]
        anchor = torch.where((length == 1)[:, None], tail, (head + tail) * 0.5)
        delta = z.sigmoid()[:, 1:] - z.sigmoid()[:, :-1]
        pairs = (mask[:, 1:] & mask[:, :-1])[:, :, None]
        denom = pairs.sum(1).clamp_min(1)
        tv = (delta.abs() * pairs).sum(1) / denom
        mono = ((-delta).relu() * pairs).sum(1) / denom
        return head_mean(anchor + self.tv * tv + self.mono * mono, valid)
