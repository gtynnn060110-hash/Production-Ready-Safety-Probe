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


class SwimDynamicWeighted(nn.Module):
    """SWiM token-level BCE with EMA-based balancing across task heads."""

    def __init__(
        self,
        window_size: int = 16,
        temperature: float = 1.0,
        loss_ema_rho: float = 0.9,
        head_weight_warmup_steps: int = 3000,
        head_weight_min: float = 0.5,
        head_weight_max: float = 2.0,
    ):
        super().__init__()
        if type(window_size) is not int or window_size < 1:
            raise ValueError("window_size must be a positive integer")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not 0 <= loss_ema_rho < 1:
            raise ValueError("loss_ema_rho must be in [0, 1)")
        if type(head_weight_warmup_steps) is not int or head_weight_warmup_steps < 0:
            raise ValueError("head_weight_warmup_steps must be a non-negative integer")
        if (
            not math.isfinite(head_weight_min)
            or not math.isfinite(head_weight_max)
            or head_weight_min <= 0
            or head_weight_max < head_weight_min
        ):
            raise ValueError("head weight bounds must be positive and ordered")
        self.window_size = window_size
        self.temperature = temperature
        self.loss_ema_rho = loss_ema_rho
        self.head_weight_warmup_steps = head_weight_warmup_steps
        self.head_weight_min = head_weight_min
        self.head_weight_max = head_weight_max
        self.register_buffer("head_loss_ema", torch.empty(0), persistent=True)
        self.register_buffer("head_loss_seen", torch.empty(0, dtype=torch.bool), persistent=True)
        self.register_buffer("pos_weight", torch.empty(0), persistent=True)
        self.register_buffer("step_count", torch.zeros((), dtype=torch.long), persistent=True)
        self.diagnostics = {}

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for name in ("head_loss_ema", "head_loss_seen", "pos_weight"):
            key = prefix + name
            if key in state_dict and self._buffers[name].shape != state_dict[key].shape:
                self._buffers[name] = torch.empty_like(state_dict[key])
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def set_pos_weight(self, labels, label_mask, cap: float = 50.0):
        labels, label_mask = labels.float(), label_mask.bool()
        positive = ((labels > 0.5) & label_mask).sum(0).float()
        total = label_mask.sum(0).float()
        negative = total - positive
        self.pos_weight = (negative / positive.clamp_min(1)).clamp(max=cap).cpu()

    def _per_sample_head_loss(self, output, labels, label_mask):
        logits = output.local_logits.float()
        token_mask = output.local_mask.bool()
        num_heads = labels.shape[1]
        sample_losses = torch.zeros(
            (logits.shape[0], num_heads), dtype=logits.dtype, device=logits.device
        )
        for row in range(logits.shape[0]):
            positions = token_mask[row].nonzero(as_tuple=True)[0]
            if positions.numel() == 0:
                continue
            sequence = logits[row, positions]
            if sequence.shape[0] >= self.window_size:
                windows = sequence.unfold(0, self.window_size, 1).transpose(1, 2)
                smoothed = windows.mean(dim=1)
            else:
                smoothed = sequence.mean(dim=0, keepdim=True)
            targets = labels[row].float().unsqueeze(0).expand_as(smoothed)
            pos_weight = self.pos_weight.to(device=smoothed.device) if self.pos_weight.numel() else None
            bce = F.binary_cross_entropy_with_logits(
                smoothed, targets, pos_weight=pos_weight, reduction="none"
            )
            weights = torch.softmax(smoothed / self.temperature, dim=0)
            sample_losses[row] = (weights * bce).sum(dim=0)
        valid = label_mask.bool() & token_mask.any(1, keepdim=True)
        counts = valid.sum(0)
        per_head = (sample_losses * valid).sum(0) / counts.clamp_min(1)
        return per_head, counts > 0

    def forward(self, output: ProbeOutput, labels, label_mask):
        per_head, valid = self._per_sample_head_loss(output, labels, label_mask)
        if self.head_loss_ema.numel() == 0:
            self.head_loss_ema = torch.zeros(labels.shape[1], dtype=torch.float32)
            self.head_loss_seen = torch.zeros(labels.shape[1], dtype=torch.bool)
        if self.head_loss_ema.numel() != labels.shape[1]:
            raise ValueError("Loss head state does not match the label head count")

        detached = per_head.detach().float().cpu()
        valid_cpu = valid.detach().cpu()
        updated = self.loss_ema_rho * self.head_loss_ema + (1 - self.loss_ema_rho) * detached
        updated = torch.where(self.head_loss_seen, updated, detached)
        self.head_loss_ema.copy_(torch.where(valid_cpu, updated, self.head_loss_ema))
        self.head_loss_seen |= valid_cpu
        self.step_count.add_(1)

        if int(self.step_count) <= self.head_weight_warmup_steps:
            weights = torch.ones_like(self.head_loss_ema)
            mode = "equal_warmup"
        else:
            active = self.head_loss_seen
            mean = self.head_loss_ema[active].mean().clamp_min(1e-8)
            weights = torch.ones_like(self.head_loss_ema)
            weights[active] = (self.head_loss_ema[active] / mean).clamp(
                self.head_weight_min, self.head_weight_max
            )
            mode = "dynamic"
        weights = weights.to(device=per_head.device, dtype=per_head.dtype)
        self.diagnostics = {
            "head_weight_mode": mode,
            "head_weights": weights.detach().cpu().tolist(),
        }
        return (per_head * weights * valid.to(per_head.dtype)).sum()


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
