"""Readouts and validation-only calibration, using logits to avoid saturation ties."""

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from .contracts import HEADS, ProbeOutput


def readout(output: ProbeOutput, mode: str, ema_alpha: float = 0.5):
    z, mask = output.local_logits.float(), output.local_mask
    if not mask.any(1).all():
        raise ValueError("Cannot read out an empty response")
    if mode == "mean":
        return z.masked_fill(~mask[:, :, None], 0).sum(1) / mask.sum(1)[:, None]
    if mode == "max":
        return z.masked_fill(~mask[:, :, None], -torch.inf).amax(1)
    if mode == "last":
        index = torch.arange(z.shape[1], device=z.device)[None].expand_as(mask)
        index = index.masked_fill(~mask, -1).amax(1)
        return z[torch.arange(z.shape[0], device=z.device), index]
    if mode == "ema_max":
        if not 0 < ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0, 1]")
        probabilities = z.sigmoid()
        ema = torch.zeros_like(probabilities[:, 0])
        maxima = torch.zeros_like(ema)
        seen = torch.zeros((z.shape[0], 1), dtype=torch.bool, device=z.device)
        for position in range(z.shape[1]):
            current_valid = mask[:, position, None]
            updated = ema_alpha * probabilities[:, position] + (1 - ema_alpha) * ema
            updated = torch.where(seen, updated, probabilities[:, position])
            ema = torch.where(current_valid, updated, ema)
            maxima = torch.where(current_valid, torch.maximum(maxima, ema), maxima)
            seen |= current_valid
        return maxima
    raise ValueError(f"Unknown readout: {mode}")


def best_f1_threshold(y, score):
    y, score = np.asarray(y, dtype=int), np.asarray(score, dtype=np.float64)
    if not len(y):
        return None
    if not y.sum():
        return float(np.nextafter(score.max(), np.inf))
    precision, recall, thresholds = precision_recall_curve(y, score)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-30)
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])


def fpr_threshold(y, score, target):
    negatives = np.sort(np.asarray(score, dtype=np.float64)[np.asarray(y) == 0])[::-1]
    if not len(negatives):
        return None
    allowed = int(np.floor(target * len(negatives)))
    return float(np.nextafter(negatives[allowed], np.inf))


def calibrate(labels, label_mask, scores, fpr_targets):
    result = {}
    for c, head in enumerate(HEADS):
        valid = label_mask[:, c].astype(bool)
        y, z = labels[valid, c], scores[valid, c].astype(np.float64)
        result[head] = {
            "f1": best_f1_threshold(y, z),
            "fpr": {str(f): fpr_threshold(y, z, f) for f in fpr_targets},
            "validation_n": len(y),
        }
    return result


def binary_metrics(y, score, threshold, *, score_type="logit"):
    y, score = np.asarray(y, dtype=bool), np.asarray(score, dtype=np.float64)
    if not np.isfinite(score).all():
        raise ValueError("Non-finite prediction logits")
    n, positives = len(y), int(y.sum())
    result = {
        "n": n,
        "positives": positives,
        "negatives": n - positives,
        "threshold_score": threshold,
        "score_type": score_type,
        "auroc": float(roc_auc_score(y, score)) if 0 < positives < n else None,
        "auprc": float(average_precision_score(y, score)) if positives else None,
    }
    result["threshold_probability" if score_type == "probability" else "threshold_logit"] = threshold
    if not n or threshold is None:
        return result | dict.fromkeys(["acc", "precision", "recall", "f1", "fpr"])
    predicted = score >= threshold
    tp, fp = int((predicted & y).sum()), int((predicted & ~y).sum())
    fn, tn = positives - tp, n - positives - fp
    return result | {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "acc": (tp + tn) / n,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / positives if positives else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "fpr": fp / (n - positives) if n - positives else None,
    }


def evaluate_scores(
    labels,
    label_mask,
    scores,
    thresholds,
    *,
    threshold_metric="f1",
    fpr_target=None,
    score_type="logit",
):
    if threshold_metric not in {"f1", "fpr"}:
        raise ValueError("Unknown primary threshold metric")
    if threshold_metric == "fpr" and fpr_target is None:
        raise ValueError("FPR evaluation requires a target")
    result = {}
    for c, head in enumerate(HEADS):
        valid = label_mask[:, c].astype(bool)
        y, z = labels[valid, c], scores[valid, c]
        threshold = (
            thresholds[head]["f1"]
            if threshold_metric == "f1"
            else thresholds[head]["fpr"][str(fpr_target)]
        )
        result[head] = binary_metrics(y, z, threshold, score_type=score_type)
        result[head]["at_validation_fpr"] = {
            target: binary_metrics(y, z, threshold, score_type=score_type)
            for target, threshold in thresholds[head]["fpr"].items()
        }
    return result
