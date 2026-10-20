#!/usr/bin/env python3
"""Score BeaverTails validation once for FPR calibration of four probes.

The expensive Qwen hidden-state forward pass is shared across checkpoints;
each probe then computes its full-response score from that same activation.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.probe_runtime import DEFAULT_MODEL, extract_hidden, load_backbone, read_jsonl  # noqa: E402
from train.probe_models import (  # noqa: E402
    HEAD_NAMES,
    CausalAttentionProbe,
    MaxOfRollingMeansProbe,
    ema_max_probabilities,
    targets_for_row,
)


def load_probe(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    architecture = checkpoint.get("architecture", "max_of_rolling_means")
    if architecture == "max_of_rolling_means":
        probe = MaxOfRollingMeansProbe(
            int(checkpoint["hidden_size"]),
            int(checkpoint["window_size"]),
            num_heads=len(checkpoint.get("head_names", HEAD_NAMES)),
        )
    elif architecture == "causal_attention_swim_v1":
        probe = CausalAttentionProbe(
            int(checkpoint["hidden_size"]),
            mlp_hidden_size=int(checkpoint.get("mlp_hidden_size", 256)),
            attention_heads=int(checkpoint.get("attention_heads", 8)),
            num_tasks=int(checkpoint.get("num_tasks", 4)),
            dropout=float(checkpoint.get("dropout", 0.1)),
        )
    else:
        raise ValueError(f"Unsupported architecture {architecture!r} in {checkpoint_path}")
    probe.load_state_dict(checkpoint["state_dict"])
    probe.to(device).eval()
    return probe, checkpoint, architecture


@torch.inference_mode()
def get_full_response_probabilities(probe, checkpoint, architecture, hidden, mask):
    if architecture == "max_of_rolling_means":
        logits = probe(hidden, mask).reshape(-1).float()
        return torch.sigmoid(logits).cpu().tolist()
    token_logits = probe(hidden, mask)
    probabilities = ema_max_probabilities(
        token_logits, mask, float(checkpoint.get("ema_alpha", 0.5))
    )
    return probabilities.reshape(-1).float().cpu().tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rolling-checkpoint", type=Path, default=ROOT / "outputs/rolling_probe_4head.pt")
    parser.add_argument("--attention-1epoch-checkpoint", type=Path,
                        default=ROOT / "outputs/attention_swim_probe_4head.pt")
    parser.add_argument("--attention-2epoch-checkpoint", type=Path,
                        default=ROOT / "outputs/attention_swim_probe_2epoch_warmup6000.pt")
    parser.add_argument("--attention-stable-checkpoint", type=Path,
                        default=ROOT / "outputs/attention_swim_probe_equal_weights_1epoch.pt")
    parser.add_argument("--validation", type=Path,
                        default=ROOT / "data/processed/beavertails_v/validation.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--model", type=Path, default=None, help="Defaults to checkpoint backbone path")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional smoke-test cap")
    args = parser.parse_args()

    checkpoints = {
        "rolling_max": args.rolling_checkpoint,
        "attention_swim_1epoch_dynamic": args.attention_1epoch_checkpoint,
        "attention_swim_2epoch_dynamic": args.attention_2epoch_checkpoint,
        "attention_swim_1epoch_stable_weight": args.attention_stable_checkpoint,
    }
    for name, path in checkpoints.items():
        if not path.is_file():
            raise SystemExit(f"Missing {name} checkpoint: {path}")
    loaded = {name: torch.load(path, map_location="cpu", weights_only=True)
              for name, path in checkpoints.items()}
    first = next(iter(loaded.values()))
    model_path = args.model or Path(first.get("model_path", DEFAULT_MODEL))
    layer = int(first.get("layer", -1))
    for name, checkpoint in loaded.items():
        candidate_path = Path(checkpoint.get("model_path", model_path))
        if candidate_path != model_path:
            raise SystemExit(f"{name} uses {candidate_path}; expected shared backbone {model_path}")
        if int(checkpoint.get("layer", -1)) != layer:
            raise SystemExit(f"{name} uses layer {checkpoint.get('layer')}; expected layer {layer}")

    processor, backbone, device = load_backbone(model_path)
    probes = {}
    for name, path in checkpoints.items():
        probe, checkpoint, architecture = load_probe(path, device)
        probes[name] = (probe, checkpoint, architecture)

    rows = read_jsonl(args.validation)
    if args.max_rows is not None:
        rows = rows[:args.max_rows]
    if not rows:
        raise SystemExit(f"No validation rows in {args.validation}")

    fields = ["id", *(f"reference_label_{head}" for head in HEAD_NAMES),
              *(f"probe_probability_{head}" for head in HEAD_NAMES)]
    outputs = {}
    writers = {}
    streams = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for name in probes:
            path = args.output_dir / f"beavertails_validation_{name}_full.csv"
            stream = path.open("w", newline="", encoding="utf-8")
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            outputs[name] = path
            writers[name] = writer
            streams[name] = stream

        for index, row in enumerate(rows, 1):
            hidden, mask = extract_hidden(processor, backbone, device, row, layer)
            targets = targets_for_row(row).tolist()
            for name, (probe, checkpoint, architecture) in probes.items():
                probabilities = get_full_response_probabilities(
                    probe, checkpoint, architecture, hidden, mask
                )
                result = {"id": row["id"]}
                result.update({f"reference_label_{head}": int(targets[i])
                               for i, head in enumerate(HEAD_NAMES)})
                result.update({f"probe_probability_{head}": probabilities[i]
                               for i, head in enumerate(HEAD_NAMES)})
                writers[name].writerow(result)
            if index % 50 == 0 or index == len(rows):
                for stream in streams.values():
                    stream.flush()
                print(f"validation {index}/{len(rows)}", flush=True)
    finally:
        for stream in streams.values():
            stream.close()

    print("Calibration score CSVs:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
