#!/usr/bin/env python3
"""Train the causal attention + SWiM multi-head probe on BeaverTails-V."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.probe_runtime import DEFAULT_MODEL, extract_hidden, load_backbone, read_jsonl  # noqa: E402
from train.probe_models import (  # noqa: E402
    HEAD_NAMES,
    CausalAttentionProbe,
    ema_max_probabilities,
    swim_logits,
    targets_for_row,
)
from train.train_rolling_probe import cap_rows  # noqa: E402


def tune_probability_threshold(labels: list[int], probabilities: list[float]) -> float:
    if len(set(labels)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(f1.argmax())])


def task_losses(
    token_logits: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
    pos_weight: torch.Tensor,
    window_size: int,
    temperature: float,
) -> torch.Tensor:
    smoothed, valid = swim_logits(token_logits, mask, window_size)
    # BCE is computed independently at each eligible smoothed token position.
    targets = target[:, None, :].expand_as(smoothed)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        smoothed, targets, pos_weight=pos_weight, reduction="none"
    )
    scores = smoothed / temperature
    scores = scores.masked_fill(~valid.unsqueeze(-1), torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=1)
    return (weights * bce).sum(dim=1).mean(dim=0)


def classification_metrics(labels: list[int], probabilities: list[float]) -> dict[str, float | None]:
    predictions = [value >= 0.5 for value in probabilities]
    result: dict[str, float | None] = {
        "auroc": None,
        "auprc": None,
        "f1_at_0_5": float(f1_score(labels, predictions, zero_division=0)),
    }
    if len(set(labels)) == 2:
        result["auroc"] = float(roc_auc_score(labels, probabilities))
        result["auprc"] = float(average_precision_score(labels, probabilities))
    return result


def validate(rows, processor, backbone, device, probe, layer, window_size, ema_alpha):
    labels = {name: [] for name in HEAD_NAMES}
    scores = {name: [] for name in HEAD_NAMES}
    probe.eval()
    for index, row in enumerate(rows, 1):
        hidden, mask = extract_hidden(processor, backbone, device, row, layer)
        with torch.no_grad():
            token_logits = probe(hidden, mask)
            probabilities = ema_max_probabilities(token_logits, mask, ema_alpha).squeeze(0).float().cpu().tolist()
        targets = targets_for_row(row).tolist()
        for head_index, name in enumerate(HEAD_NAMES):
            labels[name].append(int(targets[head_index]))
            scores[name].append(float(probabilities[head_index]))
        if index % 50 == 0 or index == len(rows):
            print(f"validation {index}/{len(rows)}", flush=True)
    metrics = {name: classification_metrics(labels[name], scores[name]) for name in HEAD_NAMES}
    thresholds = {name: tune_probability_threshold(labels[name], scores[name]) for name in HEAD_NAMES}
    return metrics, thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Keep the baseline trainer's shared CLI arguments and defaults/API names.
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--train", type=Path, default=ROOT / "data/processed/beavertails_v/train.jsonl")
    parser.add_argument("--validation", type=Path, default=ROOT / "data/processed/beavertails_v/validation.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/attention_swim_probe.pt")
    parser.add_argument("--layer", type=int, default=-1, help="Hidden-state tuple index; -1 is the final layer")
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-train-rows", type=int, default=None, help="Cap rows for a quick smoke run")
    parser.add_argument("--max-validation-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    # New architecture/objective parameters.
    parser.add_argument("--attention-heads", type=int, default=8, help="Internal attention subheads per output task")
    parser.add_argument("--mlp-hidden-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0, help="SWiM token-softmax temperature")
    parser.add_argument("--ema-alpha", type=float, default=0.5, help="Inference token-score EMA coefficient")
    parser.add_argument("--loss-ema-rho", type=float, default=0.9, help="Training-time task-loss EMA coefficient")
    parser.add_argument(
        "--head-weight-warmup-steps", type=int, default=3000,
        help="Use equal task weights for this many optimizer steps before dynamic weighting",
    )
    parser.add_argument("--head-weight-min", type=float, default=0.5)
    parser.add_argument("--head-weight-max", type=float, default=2.0)
    args = parser.parse_args()
    if args.epochs < 1 or args.window_size < 1 or args.temperature <= 0:
        parser.error("epochs/window-size must be positive and temperature must be > 0")
    if not 0 <= args.loss_ema_rho < 1 or not 0 < args.ema_alpha <= 1:
        parser.error("loss-ema-rho must be in [0, 1), and ema-alpha must be in (0, 1]")
    if args.head_weight_warmup_steps < 0:
        parser.error("head-weight-warmup-steps must be non-negative")
    if args.head_weight_min <= 0 or args.head_weight_max < args.head_weight_min:
        parser.error("head-weight bounds must be positive and max >= min")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_rows = [row for row in read_jsonl(args.train) if row.get("label") in (0, 1)]
    validation_rows = [row for row in read_jsonl(args.validation) if row.get("label") in (0, 1)]
    train_rows = cap_rows(train_rows, args.max_train_rows, args.seed)
    validation_rows = cap_rows(validation_rows, args.max_validation_rows, args.seed + 1)
    if not train_rows or not validation_rows:
        raise SystemExit("Train and validation splits must both contain labeled records.")
    random.Random(args.seed).shuffle(train_rows)
    print(
        f"Rows: train={len(train_rows)} validation={len(validation_rows)}; "
        f"labels={sum(row['label'] for row in train_rows)}/{len(train_rows)} unsafe"
    )
    train_targets = torch.stack([targets_for_row(row) for row in train_rows])
    positive_counts = train_targets.sum(dim=0)
    negative_counts = len(train_rows) - positive_counts
    pos_weight = (negative_counts / positive_counts.clamp_min(1)).clamp(max=50.0)
    print("Positive examples by head: " + json.dumps({name: int(positive_counts[i]) for i, name in enumerate(HEAD_NAMES)}))

    processor, backbone, device = load_backbone(args.model)
    probe: CausalAttentionProbe | None = None
    optimizer = None
    loss_ema: torch.Tensor | None = None
    final_metrics = {}
    thresholds = {}
    last_weights = torch.ones(len(HEAD_NAMES), dtype=torch.float32)
    optimizer_step = 0

    for epoch in range(args.epochs):
        total_loss = 0.0
        for index, row in enumerate(train_rows, 1):
            hidden, mask = extract_hidden(processor, backbone, device, row, args.layer)
            if probe is None:
                probe = CausalAttentionProbe(
                    hidden.shape[-1],
                    mlp_hidden_size=args.mlp_hidden_size,
                    attention_heads=args.attention_heads,
                    num_tasks=len(HEAD_NAMES),
                    dropout=args.dropout,
                ).to(hidden.device)
                optimizer = torch.optim.AdamW(
                    probe.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
                )
            probe.train()
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            target = targets_for_row(row).unsqueeze(0).to(device=hidden.device)
            head_loss = task_losses(
                probe(hidden, mask), mask, target,
                pos_weight.to(device=hidden.device, dtype=hidden.dtype),
                args.window_size, args.temperature,
            )
            detached = head_loss.detach().float()
            loss_ema = detached if loss_ema is None else args.loss_ema_rho * loss_ema + (1 - args.loss_ema_rho) * detached
            optimizer_step += 1
            if optimizer_step <= args.head_weight_warmup_steps:
                # Keep the objective stationary while collecting a smoothed
                # per-task loss history for the later dynamic-weight phase.
                last_weights = torch.ones_like(loss_ema)
            else:
                raw_weights = loss_ema / loss_ema.mean().clamp_min(1e-8)
                last_weights = raw_weights.clamp(args.head_weight_min, args.head_weight_max)
            loss = (head_loss * last_weights.to(dtype=head_loss.dtype)).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach())
            if index % 50 == 0 or index == len(train_rows):
                print(
                    f"epoch {epoch + 1}/{args.epochs} train {index}/{len(train_rows)} "
                    f"step={optimizer_step} loss={total_loss / index:.4f} "
                    f"weight_mode={'equal_warmup' if optimizer_step <= args.head_weight_warmup_steps else 'dynamic'} "
                    f"head_weights="
                    f"{json.dumps({name: round(float(last_weights[i]), 3) for i, name in enumerate(HEAD_NAMES)})}",
                    flush=True,
                )

        assert probe is not None
        final_metrics, thresholds = validate(
            validation_rows, processor, backbone, device, probe, args.layer, args.window_size, args.ema_alpha
        )
        print(f"epoch {epoch + 1} validation={json.dumps(final_metrics)} thresholds={json.dumps(thresholds)}")

    assert probe is not None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "architecture": "causal_attention_swim_v1",
        "state_dict": probe.state_dict(),
        "hidden_size": probe.hidden_size,
        "mlp_hidden_size": probe.mlp_hidden_size,
        "attention_heads": probe.attention_heads,
        "num_tasks": probe.num_tasks,
        "head_names": list(HEAD_NAMES),
        "window_size": args.window_size,
        "temperature": args.temperature,
        "ema_alpha": args.ema_alpha,
        "layer": args.layer,
        "thresholds": thresholds,
        "model_path": str(args.model),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "loss_ema_rho": args.loss_ema_rho,
        "head_weight_warmup_steps": args.head_weight_warmup_steps,
        "final_task_loss_ema": None if loss_ema is None else loss_ema.cpu().tolist(),
        "final_task_weights": last_weights.cpu().tolist(),
        "seed": args.seed,
        "validation_metrics": final_metrics,
        "pos_weight": {name: float(pos_weight[i]) for i, name in enumerate(HEAD_NAMES)},
        "target": "three one-vs-rest response-risk heads plus overall response safety; 1=unsafe, 0=safe",
        "category_policy": "Only the matching risk head is positive for unsafe BeaverTails-V records; other risk heads are explicit negatives; overall_safety uses every response label.",
        "inference_aggregation": "max over causal per-token sigmoid-logit EMA probabilities; EMA initialized from first valid token",
    }
    torch.save(checkpoint, args.output)
    print(f"Saved probe checkpoint: {args.output}")


if __name__ == "__main__":
    main()
