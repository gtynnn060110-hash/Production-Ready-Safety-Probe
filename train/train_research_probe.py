#!/usr/bin/env python3
"""Train text-eval MQA-RMAttn or SingProbe on BeaverTails-V image records."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn.utils.rnn import pad_sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.probe_runtime import DEFAULT_MODEL, extract_hidden, load_backbone, read_jsonl  # noqa: E402
from train.probe_models import HEAD_NAMES, targets_for_row  # noqa: E402
from train.research_probe_models import (  # noqa: E402
    MQARMAttnProbe,
    SingProbe,
    max_readout,
    singprobe_weighted_loss,
)


def atomic_torch_save(payload: dict, path: Path) -> None:
    """Write a checkpoint atomically so a failed save cannot truncate its predecessor."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp_path.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def atomic_write_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def make_probe(name: str, hidden_size: int, context_ffn_hidden_size: int = 0) -> torch.nn.Module:
    if name == "mqa_rmattn":
        return MQARMAttnProbe(hidden_size, num_tasks=len(HEAD_NAMES), context_ffn_hidden_size=context_ffn_hidden_size)
    if name == "singprobe":
        return SingProbe(hidden_size, num_tasks=len(HEAD_NAMES))
    raise ValueError(f"Unknown probe: {name}")


def collate_hidden(rows, processor, backbone, device, layer, norm_eps):
    hidden_states = []
    targets = []
    for row in rows:
        hidden, mask = extract_hidden(processor, backbone, device, row, layer)
        hidden = hidden.float()
        hidden = hidden * torch.rsqrt(hidden.square().mean(dim=-1, keepdim=True) + norm_eps)
        hidden_states.append(hidden[0].to(device=device, dtype=torch.float32))
        targets.append(targets_for_row(row))
    padded = pad_sequence(hidden_states, batch_first=True)
    lengths = torch.tensor([x.shape[0] for x in hidden_states], device=device)
    mask = torch.arange(padded.shape[1], device=device)[None, :] < lengths[:, None]
    return padded, mask, torch.stack(targets).to(device=device)


def thresholds_at_fpr(labels: np.ndarray, scores: np.ndarray, target: float) -> dict[str, float]:
    result = {}
    for head_index, head in enumerate(HEAD_NAMES):
        y = labels[:, head_index].astype(bool)
        negatives = np.sort(scores[~y, head_index])[::-1]
        if not len(negatives):
            result[head] = float("inf")
            continue
        allowed_false_positives = int(math.floor(target * len(negatives)))
        threshold = np.nextafter(negatives[allowed_false_positives], np.inf)
        result[head] = float(threshold)
    return result


def evaluate_validation(rows, processor, backbone, device, probe, layer, batch_size, fpr_target, norm_eps):
    labels, scores = [], []
    probe.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            hidden, mask, targets = collate_hidden(
                batch_rows, processor, backbone, device, layer, norm_eps=norm_eps
            )
            token_logits = probe(hidden, mask)
            sequence_logits = max_readout(token_logits, mask)
            probabilities = sequence_logits.sigmoid().cpu().numpy()
            labels.extend(targets.cpu().numpy())
            scores.extend(probabilities)
            done = min(start + len(batch_rows), len(rows))
            if done % 100 == 0 or done == len(rows):
                print(f"validation {done}/{len(rows)}", flush=True)

    labels_np = np.asarray(labels, dtype=np.float32)
    scores_np = np.asarray(scores, dtype=np.float64)
    metrics = {}
    for i, head in enumerate(HEAD_NAMES):
        y, s = labels_np[:, i], scores_np[:, i]
        predictions = s >= 0.5
        metrics[head] = {
            "n": int(len(y)),
            "positives": int(y.sum()),
            "auroc": float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None,
            "auprc": float(average_precision_score(y, s)) if y.sum() else None,
            "f1_at_0_5": float(
                2 * np.logical_and(predictions, y == 1).sum()
                / max(1, predictions.sum() + (y == 1).sum())
            ),
        }
    thresholds = thresholds_at_fpr(labels_np, scores_np, fpr_target)
    return metrics, thresholds, labels_np, scores_np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", choices=("mqa_rmattn", "singprobe"), required=True)
    parser.add_argument("--context-ffn-hidden-size", type=int, default=0, help="Add residual context FFN to MQA-RMAttn; 0 disables it")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--train", type=Path, default=ROOT / "data/processed/beavertails_v/train.jsonl")
    parser.add_argument("--validation", type=Path, default=ROOT / "data/processed/beavertails_v/validation.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=15, help="Zero-based Qwen hidden-state tuple index")
    parser.add_argument("--norm-eps", type=float, default=1e-6, help="RMS no-affine feature normalization epsilon")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--threshold-fpr", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--beta-tau", type=float, default=0.35)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-validation-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.context_ffn_hidden_size < 0 or (args.context_ffn_hidden_size and args.probe != "mqa_rmattn"):
        parser.error("context FFN requires mqa_rmattn and a non-negative size")
    if args.epochs < 1 or args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        parser.error("epochs, batch-size and gradient-accumulation-steps must be positive")
    if not 0 < args.threshold_fpr < 1 or args.temperature <= 0 or args.beta_tau <= 0 or args.norm_eps <= 0:
        parser.error("threshold-fpr must be in (0,1), and loss parameters must be positive")
    learning_rate = args.learning_rate if args.learning_rate is not None else (1e-4 if args.probe == "mqa_rmattn" else 3e-4)
    if learning_rate <= 0:
        parser.error("learning-rate must be positive")
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}; choose a new path to avoid overwriting it.")
    epoch_outputs = list(args.output.parent.glob(f"{args.output.stem}.epoch*{args.output.suffix}"))
    if epoch_outputs:
        raise SystemExit(
            f"Epoch checkpoints already exist for {args.output}; choose a new --output path."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_rows = [r for r in read_jsonl(args.train) if r.get("label") in (0, 1)]
    validation_rows = [r for r in read_jsonl(args.validation) if r.get("label") in (0, 1)]
    if args.max_train_rows is not None:
        train_rows = train_rows[: args.max_train_rows]
    if args.max_validation_rows is not None:
        validation_rows = validation_rows[: args.max_validation_rows]
    if not train_rows or not validation_rows:
        raise SystemExit("Train and validation splits must both contain labeled records.")

    print(
        f"Probe={args.probe} rows: train={len(train_rows)} validation={len(validation_rows)} "
        f"epochs={args.epochs} batch={args.batch_size} accumulation={args.gradient_accumulation_steps} "
        f"lr={learning_rate:g} weighted_loss=SingProbeWeighted",
        flush=True,
    )

    processor, backbone, device = load_backbone(args.model)
    # Build lazily after seeing the selected backbone's hidden width.
    first_hidden, _ = extract_hidden(processor, backbone, device, train_rows[0], args.layer)
    probe = make_probe(args.probe, first_hidden.shape[-1], args.context_ffn_hidden_size).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    batches_per_epoch = math.ceil(len(train_rows) / args.batch_size)
    updates_per_epoch = math.ceil(batches_per_epoch / args.gradient_accumulation_steps)
    total_updates = args.epochs * updates_per_epoch
    warmup_updates = int(total_updates * args.warmup_ratio)

    def lr_scale(step: int) -> float:
        if warmup_updates and step < warmup_updates:
            return (step + 1) / warmup_updates
        progress = min(1.0, max(0.0, (step - warmup_updates) / max(1, total_updates - warmup_updates)))
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    history = []
    best_overall_auprc = -1.0
    best_state = None
    final_thresholds = {}
    probe_config = (
        {
            "num_query_heads": 4, "num_kv_heads": 1, "head_dim": 64,
            "context_window": 0, "risk_window": 40, "rms_norm_eps": 1e-6,
            "attention_chunk_size": 256, "mlp_hidden_sizes": [100, 100],
            "internal_heads": 10, "context_ffn_hidden_size": args.context_ffn_hidden_size,
        }
        if args.probe == "mqa_rmattn"
        else {
            "num_query_heads": 4, "num_kv_heads": 1, "head_dim": 64,
            "sliding_window": 0, "rms_norm_eps": 1e-6,
            "init_bias": 0.0, "attention_chunk_size": 256,
        }
    )

    def checkpoint_payload(state_dict, thresholds, completed_epochs, best_epoch):
        return {
            "architecture": "mqa_rmattn_v1" if args.probe == "mqa_rmattn" else "singprobe_v1",
            "probe_name": args.probe,
            "state_dict": state_dict,
            "hidden_size": first_hidden.shape[-1],
            "head_names": list(HEAD_NAMES),
            "num_tasks": len(HEAD_NAMES),
            "layer": args.layer,
            "normalize": "rms_no_affine",
            "norm_eps": args.norm_eps,
            "thresholds": thresholds,
            "threshold_metric": "fpr",
            "threshold_fpr": args.threshold_fpr,
            "model_path": str(args.model),
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "epochs": args.epochs,
            "completed_epochs": completed_epochs,
            "best_epoch": best_epoch,
            "learning_rate": learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "temperature": args.temperature,
            "beta_tau": args.beta_tau,
            "probe_config": probe_config,
            "loss_config": {
                "name": "singprobe_weighted",
                "temperature": args.temperature,
                "beta_tau": args.beta_tau,
                "detach_weights": True,
            },
            "inference_aggregation": "max over token logits, then sigmoid; all multimodal sequence tokens are valid",
            "history": list(history),
        }

    checkpoint = None
    for epoch in range(args.epochs):
        rng = random.Random(args.seed + epoch)
        order = list(range(len(train_rows)))
        rng.shuffle(order)
        probe.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_losses = []
        groups = [order[i : i + args.batch_size] for i in range(0, len(order), args.batch_size)]
        for batch_index, indices in enumerate(groups):
            batch_rows = [train_rows[i] for i in indices]
            hidden, mask, batch_targets = collate_hidden(
                batch_rows, processor, backbone, device, args.layer, args.norm_eps
            )
            logits = probe(hidden, mask)
            loss = singprobe_weighted_loss(
                logits, mask, batch_targets,
                temperature=args.temperature, beta_tau=args.beta_tau, detach_weights=True,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch + 1}, batch={batch_index + 1}")
            group_start = (batch_index // args.gradient_accumulation_steps) * args.gradient_accumulation_steps
            group_size = min(args.gradient_accumulation_steps, len(groups) - group_start)
            (loss / group_size).backward()
            epoch_losses.append(float(loss.detach()))
            update_now = (
                (batch_index + 1) % args.gradient_accumulation_steps == 0
                or batch_index + 1 == len(groups)
            )
            if update_now:
                torch.nn.utils.clip_grad_norm_(probe.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            completed = batch_index + 1
            if completed % 100 == 0 or completed == len(groups):
                print(
                    f"epoch {epoch + 1}/{args.epochs} train {completed}/{len(groups)} "
                    f"loss={sum(epoch_losses) / len(epoch_losses):.5f} lr={scheduler.get_last_lr()[0]:.3g}",
                    flush=True,
                )

        val_metrics, thresholds, _, _ = evaluate_validation(
            validation_rows, processor, backbone, device, probe, args.layer,
            args.batch_size, args.threshold_fpr, args.norm_eps,
        )
        score = val_metrics["overall_safety"]["auprc"] or 0.0
        record = {
            "epoch": epoch + 1,
            "train_loss": float(sum(epoch_losses) / max(1, len(epoch_losses))),
            "validation": val_metrics,
            "thresholds_at_fpr": thresholds,
            "selection_metric": "overall_safety_auprc",
            "selection_readout": "max",
        }
        history.append(record)
        print(f"epoch {epoch + 1} summary={json.dumps(record, ensure_ascii=False)}", flush=True)
        if score > best_overall_auprc:
            best_overall_auprc = score
            final_thresholds = thresholds
            best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}

        best_epoch = max(
            range(len(history)),
            key=lambda i: history[i]["validation"]["overall_safety"]["auprc"] or 0,
        ) + 1
        epoch_checkpoint = checkpoint_payload(
            {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()},
            thresholds,
            completed_epochs=epoch + 1,
            best_epoch=epoch + 1,
        )
        epoch_path = args.output.with_name(
            f"{args.output.stem}.epoch{epoch + 1}{args.output.suffix}"
        )
        atomic_torch_save(epoch_checkpoint, epoch_path)
        checkpoint = checkpoint_payload(
            best_state,
            final_thresholds,
            completed_epochs=epoch + 1,
            best_epoch=best_epoch,
        )
        atomic_torch_save(checkpoint, args.output)
        atomic_write_json(history, args.output.with_suffix(".history.json"))
        print(
            f"Saved epoch {epoch + 1} checkpoint: {epoch_path}; "
            f"best-so-far epoch {best_epoch}: {args.output}",
            flush=True,
        )

    assert best_state is not None
    assert checkpoint is not None
    print(f"Saved best epoch {checkpoint['best_epoch']} checkpoint: {args.output}", flush=True)


if __name__ == "__main__":
    main()
