#!/usr/bin/env python3
"""Train a frozen-backbone Max-of-Rolling-Means probe on BeaverTails-V."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.probe_runtime import DEFAULT_MODEL, extract_hidden, load_backbone, read_jsonl  # noqa: E402
from train.probe_models import HEAD_CATEGORIES, HEAD_NAMES, MaxOfRollingMeansProbe, targets_for_row  # noqa: E402


def cap_rows(rows: list[dict], limit: int | None, seed: int) -> list[dict]:
    if limit is None or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    by_stratum: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        category = row.get("category")
        domain = next((name for name, values in HEAD_CATEGORIES.items() if category in values), "other")
        by_stratum.setdefault((domain, int(row["label"])), []).append(row)
    for values in by_stratum.values():
        rng.shuffle(values)
    selected = []
    strata = sorted(by_stratum)
    while len(selected) < limit and any(by_stratum.values()):
        for stratum in strata:
            if by_stratum[stratum] and len(selected) < limit:
                selected.append(by_stratum[stratum].pop())
    return selected


def metrics(labels: list[int], scores: list[float]) -> dict[str, float | None]:
    probabilities = torch.tensor(scores).sigmoid().numpy()
    result: dict[str, float | None] = {
        "auroc": None,
        "auprc": None,
        "f1_at_0_5": float(f1_score(labels, probabilities >= 0.5, zero_division=0)),
    }
    if len(set(labels)) == 2:
        result["auroc"] = float(roc_auc_score(labels, probabilities))
        result["auprc"] = float(average_precision_score(labels, probabilities))
    return result


def tune_threshold(labels: list[int], scores: list[float]) -> float:
    if len(set(labels)) < 2:
        return 0.5
    probabilities = torch.tensor(scores).sigmoid().numpy()
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(f1.argmax())])


def score_rows(rows, processor, backbone, device, probe, layer: int) -> tuple[dict[str, list[int]], dict[str, list[float]]]:
    labels = {name: [] for name in HEAD_NAMES}
    scores = {name: [] for name in HEAD_NAMES}
    probe.eval()
    for index, row in enumerate(rows, 1):
        hidden, mask = extract_hidden(processor, backbone, device, row, layer)
        with torch.no_grad():
            logits = probe(hidden, mask).squeeze(0).tolist()
        targets = targets_for_row(row).tolist()
        for head_index, name in enumerate(HEAD_NAMES):
            labels[name].append(int(targets[head_index]))
            scores[name].append(float(logits[head_index]))
        if index % 50 == 0 or index == len(rows):
            print(f"validation {index}/{len(rows)}", flush=True)
    return labels, scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--train", type=Path, default=ROOT / "data/processed/beavertails_v/train.jsonl")
    parser.add_argument("--validation", type=Path, default=ROOT / "data/processed/beavertails_v/validation.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/rolling_probe.pt")
    parser.add_argument("--layer", type=int, default=-1, help="Hidden-state tuple index; -1 is the final layer")
    parser.add_argument("--window-size", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train-rows", type=int, default=None, help="Cap rows for a quick smoke run")
    parser.add_argument("--max-validation-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_rows = [r for r in read_jsonl(args.train) if r.get("label") in (0, 1)]
    validation_rows = [r for r in read_jsonl(args.validation) if r.get("label") in (0, 1)]
    train_rows = cap_rows(train_rows, args.max_train_rows, args.seed)
    validation_rows = cap_rows(validation_rows, args.max_validation_rows, args.seed + 1)
    if not train_rows or not validation_rows:
        raise SystemExit("Train and validation splits must both contain labeled records.")
    random.Random(args.seed).shuffle(train_rows)
    print(f"Rows: train={len(train_rows)} validation={len(validation_rows)}; labels={sum(r['label'] for r in train_rows)}/{len(train_rows)} unsafe")
    train_targets = torch.stack([targets_for_row(row) for row in train_rows])
    positive_counts = train_targets.sum(dim=0)
    negative_counts = len(train_rows) - positive_counts
    pos_weight = (negative_counts / positive_counts.clamp_min(1)).clamp(max=50.0)
    print("Positive examples by head: " + json.dumps({name: int(positive_counts[i]) for i, name in enumerate(HEAD_NAMES)}))
    processor, backbone, device = load_backbone(args.model)

    probe: MaxOfRollingMeansProbe | None = None
    optimizer = None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    for epoch in range(args.epochs):
        if probe is not None:
            probe.train()
        total_loss = 0.0
        for index, row in enumerate(train_rows, 1):
            hidden, mask = extract_hidden(processor, backbone, device, row, args.layer)
            if probe is None:
                probe = MaxOfRollingMeansProbe(hidden.shape[-1], args.window_size, num_heads=len(HEAD_NAMES)).to(hidden.device)
                optimizer = torch.optim.AdamW(probe.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            probe.train()
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            target = targets_for_row(row).unsqueeze(0).to(device=hidden.device)
            loss = criterion(probe(hidden, mask), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            if index % 50 == 0 or index == len(train_rows):
                print(f"epoch {epoch + 1}/{args.epochs} train {index}/{len(train_rows)} loss={total_loss / index:.4f}", flush=True)

        assert probe is not None
        val_labels, val_scores = score_rows(validation_rows, processor, backbone, device, probe, args.layer)
        val_metrics = {name: metrics(val_labels[name], val_scores[name]) for name in HEAD_NAMES}
        thresholds = {name: tune_threshold(val_labels[name], val_scores[name]) for name in HEAD_NAMES}
        print(f"epoch {epoch + 1} validation={json.dumps(val_metrics)} thresholds={json.dumps(thresholds)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": probe.state_dict(),
        "hidden_size": probe.token_scorer.in_features,
        "head_names": list(HEAD_NAMES),
        "window_size": probe.window_size,
        "layer": args.layer,
        "thresholds": thresholds,
        "model_path": str(args.model),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "seed": args.seed,
        "validation_metrics": val_metrics,
        "pos_weight": {name: float(pos_weight[i]) for i, name in enumerate(HEAD_NAMES)},
        "target": "three one-vs-rest response-risk heads plus overall response safety; 1=unsafe, 0=safe",
        "category_policy": "Only the matching risk head is positive for unsafe BeaverTails-V records; other risk heads are explicit negatives; overall_safety uses every response label.",
    }
    torch.save(checkpoint, args.output)
    print(f"Saved probe checkpoint: {args.output}")


if __name__ == "__main__":
    main()
