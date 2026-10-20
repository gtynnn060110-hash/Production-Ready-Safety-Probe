#!/usr/bin/env python3
"""Evaluate a trained probe on BeaverTails-V or MM-SafetyBench."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.probe_runtime import (  # noqa: E402
    DEFAULT_MODEL,
    extract_hidden,
    generate_batch_with_hidden,
    judge_answer,
    load_backbone,
    read_jsonl,
)
from train.probe_models import (  # noqa: E402
    HEAD_NAMES,
    CausalAttentionProbe,
    MaxOfRollingMeansProbe,
    ema_max_probabilities,
    targets_for_row,
)


def binary_metrics(rows: list[dict], head: str, threshold: float) -> dict:
    labeled = [
        row for row in rows
        if row.get(f"judge_label_{head}") in (0, 1) or row.get(f"reference_label_{head}") in (0, 1)
    ]
    if not labeled:
        return {"n": len(rows), "labeled_n": 0}
    labels = [
        int(row[f"judge_label_{head}"] if row.get(f"judge_label_{head}") in (0, 1) else row[f"reference_label_{head}"])
        for row in labeled
    ]
    predictions = [int(row[f"probe_probability_{head}"] >= threshold) for row in labeled]
    scores = [float(row[f"probe_probability_{head}"]) for row in labeled]
    result = {
        "n": len(rows),
        "labeled_n": len(labeled),
        "unsafe_rate_reference": sum(labels) / len(labels),
        "probe_recall_at_val_threshold": float(sum(p and y for p, y in zip(predictions, labels)) / max(1, sum(labels))),
        "probe_f1_at_val_threshold": float(f1_score(labels, predictions, zero_division=0)),
        "probe_accuracy_at_val_threshold": float(accuracy_score(labels, predictions)),
    }
    if len(set(labels)) == 2:
        result["probe_auroc_vs_reference"] = float(roc_auc_score(labels, scores))
        result["probe_auprc_vs_reference"] = float(average_precision_score(labels, scores))
    else:
        result["probe_auroc_vs_reference"] = None
        result["probe_auprc_vs_reference"] = None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/rolling_probe.pt")
    parser.add_argument("--dataset", choices=("beavertails", "mm_safetybench"), default="beavertails")
    parser.add_argument("--split", choices=("official_evaluation", "validation", "train"), default="official_evaluation")
    parser.add_argument("--model", type=Path, default=None, help="Defaults to the checkpoint's backbone path")
    parser.add_argument("--max-rows", type=int, default=None, help="Cap rows for a quick smoke run")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4, help="MM-SafetyBench generation micro-batch size")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted MM-SafetyBench generation-only run from its partial CSV")
    parser.add_argument(
        "--input-csv", type=Path, default=None,
        help="Reuse saved MM-SafetyBench responses; re-extract hidden states without regenerating answers",
    )
    parser.add_argument("--scenarios", nargs="+", help="Optional MM-SafetyBench scenario names, e.g. 02-HateSpeech 04-Physical_Harm")
    parser.add_argument("--judge", action="store_true", help="Judge generated MM-SafetyBench answers via OpenAI-compatible chat-completions API")
    parser.add_argument("--judge-with-image", action="store_true", help="Send the benchmark image to a vision-capable LLM judge (requires --judge)")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.judge_with_image and not args.judge:
        parser.error("--judge-with-image requires --judge")
    if args.resume and (args.dataset != "mm_safetybench" or args.judge):
        parser.error("--resume is for MM-SafetyBench generation-only runs; run judge separately on the saved CSV")
    if args.input_csv is not None and args.dataset != "mm_safetybench":
        parser.error("--input-csv can only be used with --dataset mm_safetybench")
    if args.input_csv is not None and args.resume:
        parser.error("--resume applies only to generation runs; omit it when re-scoring an input CSV")
    if args.input_csv is not None and not args.input_csv.is_file():
        raise SystemExit(f"Input CSV not found: {args.input_csv}")

    if not args.checkpoint.is_file():
        raise SystemExit(
            f"Probe checkpoint missing: {args.checkpoint}; run the corresponding trainer first "
            "(train/train_rolling_probe.py or train/train_attention_swim_probe.py)."
        )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model_path = args.model or Path(checkpoint.get("model_path", DEFAULT_MODEL))
    if args.dataset == "beavertails":
        data_path = ROOT / f"data/processed/beavertails_v/{args.split}.jsonl"
    else:
        data_path = ROOT / "data/processed/mm_safetybench.jsonl"
    rows = read_jsonl(data_path, None if args.dataset == "mm_safetybench" else args.max_rows, args.seed)
    source_rows_by_id = {str(row.get("id", "")): row for row in rows}
    if args.dataset == "mm_safetybench" and args.scenarios:
        wanted = set(args.scenarios)
        available = {row.get("scenario") for row in rows}
        unknown = wanted - available
        if unknown:
            raise SystemExit(f"Unknown/unavailable scenarios: {sorted(unknown)}; available={sorted(available)}")
        rows = [row for row in rows if row.get("scenario") in wanted]
    cached_input_by_id: dict[str, dict] = {}
    if args.input_csv is not None:
        with args.input_csv.open(newline="", encoding="utf-8") as handle:
            cached_input_rows = list(csv.DictReader(handle))
        if not cached_input_rows:
            raise SystemExit(f"No rows in input CSV: {args.input_csv}")
        wanted_scenarios = set(args.scenarios or [])
        selected_rows = []
        for cached in cached_input_rows:
            row_id = str(cached.get("id", ""))
            if not row_id or row_id in cached_input_by_id:
                raise SystemExit("Input CSV must have non-empty, unique id values.")
            if not cached.get("response"):
                raise SystemExit(f"Missing generated response for row {row_id!r} in {args.input_csv}")
            if row_id not in source_rows_by_id:
                raise SystemExit(f"Input row {row_id!r} is absent from {data_path}")
            source_row = source_rows_by_id[row_id]
            if wanted_scenarios and source_row.get("scenario") not in wanted_scenarios:
                continue
            cached_input_by_id[row_id] = cached
            selected_rows.append(source_row)
        rows = selected_rows
        if args.max_rows is not None:
            rows = rows[:args.max_rows]
            cached_input_by_id = {str(row["id"]): cached_input_by_id[str(row["id"])] for row in rows}
    elif args.dataset == "mm_safetybench" and args.max_rows is not None and len(rows) > args.max_rows:
        random.Random(args.seed).shuffle(rows)
        rows = rows[:args.max_rows]
    if not rows:
        raise SystemExit(f"No evaluation rows found in {data_path}")
    if args.dataset == "mm_safetybench" and args.judge and not __import__("os").environ.get("OPENAI_MODEL"):
        raise SystemExit("Set OPENAI_MODEL (and usually OPENAI_API_KEY) before using --judge.")

    architecture = checkpoint.get("architecture", "max_of_rolling_means")
    is_attention_swim = architecture == "causal_attention_swim_v1"
    if architecture not in {"max_of_rolling_means", "causal_attention_swim_v1"}:
        raise SystemExit(f"Unsupported checkpoint architecture: {architecture}")
    output_stem = "attention_swim_probe" if is_attention_swim else "rolling_probe"
    output = args.output or ROOT / "outputs" / f"{args.dataset}_{args.split if args.dataset == 'beavertails' else 'test'}_{output_stem}.csv"
    generation_only = args.dataset == "mm_safetybench" and not args.judge and args.input_csv is None
    existing_results: list[dict] = []
    completed_ids: set[str] = set()
    if generation_only:
        manifest_path = output.with_suffix(".manifest.json")
        manifest = {
            "dataset": args.dataset,
            "scenarios": sorted(args.scenarios or []),
            "row_ids": [str(row.get("id", "")) for row in rows],
            "checkpoint": str(args.checkpoint.resolve()),
            "model": str(model_path.resolve()),
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "seed": args.seed,
        }
        if args.resume:
            if not output.is_file() or not manifest_path.is_file():
                raise SystemExit("Cannot resume: expected both the partial CSV and its .manifest.json file.")
            saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            # Older generation-only runs were serial and did not record a
            # batch size. Their completed rows are still valid cache entries;
            # migrate that manifest so the remaining rows can resume batched.
            legacy_manifest = {key: value for key, value in manifest.items() if key != "batch_size"}
            if saved_manifest == legacy_manifest:
                manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            elif saved_manifest != manifest:
                raise SystemExit("Resume settings do not match the original generation run; use its original options/output path.")
            with output.open(newline="", encoding="utf-8") as handle:
                existing_results = list(csv.DictReader(handle))
            requested_ids = set(manifest["row_ids"])
            for item in existing_results:
                row_id = str(item.get("id", ""))
                if row_id not in requested_ids or not item.get("response"):
                    raise SystemExit(f"Cannot resume: partial CSV has an unexpected or incomplete row: {row_id!r}")
                if row_id in completed_ids:
                    raise SystemExit(f"Cannot resume: duplicate row id in partial CSV: {row_id}")
                completed_ids.add(row_id)
        else:
            if output.exists():
                raise SystemExit(f"Output already exists: {output}; choose a new --output or pass --resume.")
            output.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    processor, backbone, device = load_backbone(model_path)
    head_names = tuple(checkpoint.get("head_names", ("overall_safety",)))
    thresholds = checkpoint.get("thresholds", {"overall_safety": checkpoint.get("threshold", 0.5)})
    if is_attention_swim:
        probe = CausalAttentionProbe(
            checkpoint["hidden_size"],
            mlp_hidden_size=checkpoint.get("mlp_hidden_size", 256),
            attention_heads=checkpoint.get("attention_heads", 8),
            num_tasks=checkpoint.get("num_tasks", len(head_names)),
        )
        ema_alpha = float(checkpoint.get("ema_alpha", 0.5))
    else:
        probe = MaxOfRollingMeansProbe(checkpoint["hidden_size"], checkpoint["window_size"], num_heads=len(head_names))
        ema_alpha = None
    probe.load_state_dict(checkpoint["state_dict"])
    probe.to(device).eval()
    layer = int(checkpoint["layer"])
    results: list[dict] = list(existing_results)
    rows_to_process = [row for row in rows if str(row.get("id", "")) not in completed_ids]
    fieldnames = ["id", "scenario", "question_id", "attack_variant", "judge_text", "response"]
    for name in head_names:
        fieldnames.extend([
            f"probe_logit_{name}", f"probe_probability_{name}", f"probe_prediction_{name}",
            f"threshold_{name}", f"judge_label_{name}", f"reference_label_{name}",
        ])
    stream = None
    writer = None
    if generation_only:
        stream = output.open("a" if args.resume else "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if not args.resume:
            writer.writeheader()
            stream.flush()
    def score_sequence(hidden: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            if is_attention_swim:
                token_logits = probe(hidden, mask)
                probabilities = ema_max_probabilities(token_logits, mask, ema_alpha).reshape(-1).float().cpu()
                # Keep the baseline CSV's logit/probability columns available.
                logits = torch.logit(probabilities.clamp(1e-7, 1 - 1e-7))
                return logits, probabilities
            logits = probe(hidden, mask).reshape(-1).float().cpu()
            return logits, torch.sigmoid(logits)

    def make_item(
        row: dict,
        response: str,
        judge_labels: dict[str, int],
        judge_text: str,
        logits: torch.Tensor,
        probabilities: torch.Tensor,
    ) -> dict:
        reference_targets = targets_for_row(row).tolist() if args.dataset == "beavertails" else None
        item = {
            "id": row.get("id", ""),
            "scenario": row.get("scenario", ""),
            "question_id": row.get("question_id", ""),
            "attack_variant": row.get("attack_variant", ""),
            "judge_text": judge_text,
            "response": response or "",
        }
        for head_index, name in enumerate(head_names):
            threshold = float(thresholds.get(name, 0.5))
            probability = float(probabilities[head_index])
            item[f"probe_logit_{name}"] = float(logits[head_index])
            item[f"probe_probability_{name}"] = probability
            item[f"probe_prediction_{name}"] = int(probability >= threshold)
            item[f"threshold_{name}"] = threshold
            item[f"judge_label_{name}"] = judge_labels.get(name)
            item[f"reference_label_{name}"] = (
                int(reference_targets[HEAD_NAMES.index(name)])
                if reference_targets is not None and name in HEAD_NAMES else None
            )
        return item

    if args.dataset == "mm_safetybench" and args.input_csv is None:
        for batch_start in range(0, len(rows_to_process), args.batch_size):
            batch_rows = rows_to_process[batch_start:batch_start + args.batch_size]
            responses, hidden, masks = generate_batch_with_hidden(
                processor, backbone, device, batch_rows, args.max_new_tokens, layer
            )
            for batch_index, row in enumerate(batch_rows):
                response = responses[batch_index]
                judge_labels: dict[str, int] = {}
                judge_text = ""
                if args.judge:
                    judge_labels, judge_text = judge_answer(row, response, include_image=args.judge_with_image)
                # Strip left-padding so each architecture scores the same
                # unpadded sequence it would see in single-example evaluation.
                valid_hidden = hidden[batch_index][masks[batch_index]].unsqueeze(0)
                valid_mask = torch.ones(valid_hidden.shape[:2], dtype=torch.bool, device=valid_hidden.device)
                logits, probabilities = score_sequence(valid_hidden, valid_mask)
                item = make_item(row, response, judge_labels, judge_text, logits, probabilities)
                results.append(item)
                if writer is not None and stream is not None:
                    writer.writerow(item)
                    stream.flush()
                complete = len(completed_ids) + batch_start + batch_index + 1
                if complete % 10 == 0 or complete == len(rows):
                    print(f"evaluated {complete}/{len(rows)}", flush=True)
            del hidden, masks
    elif args.dataset == "mm_safetybench":
        for index, row in enumerate(rows_to_process, 1):
            cached = cached_input_by_id[str(row["id"])]
            response = cached["response"]
            if args.judge:
                judge_labels, judge_text = judge_answer(row, response, include_image=args.judge_with_image)
            else:
                judge_labels = {}
                judge_text = cached.get("judge_text", "")
                for name in head_names:
                    value = cached.get(f"judge_label_{name}")
                    if value in ("0", "1", 0, 1):
                        judge_labels[name] = int(value)
            hidden, mask = extract_hidden(
                processor, backbone, device, row, layer, response_override=response
            )
            logits, probabilities = score_sequence(hidden, mask)
            item = make_item(row, response, judge_labels, judge_text, logits, probabilities)
            results.append(item)
            if index % 10 == 0 or index == len(rows_to_process):
                print(f"rescored {index}/{len(rows_to_process)} cached responses", flush=True)
    else:
        for index, row in enumerate(rows_to_process, 1):
            response = row.get("response", "")
            hidden, mask = extract_hidden(processor, backbone, device, row, layer)
            logits, probabilities = score_sequence(hidden, mask)
            item = make_item(row, response, {}, "", logits, probabilities)
            results.append(item)
            if index % 10 == 0 or index == len(rows_to_process):
                print(f"evaluated {index}/{len(rows)}", flush=True)

    if stream is not None:
        stream.close()

    if not generation_only:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    summary = {
        "dataset": args.dataset,
        "split": args.split if args.dataset == "beavertails" else "test",
        "checkpoint": str(args.checkpoint),
        "architecture": architecture,
        "input_responses_csv": str(args.input_csv) if args.input_csv is not None else None,
        "heads": list(head_names),
        "thresholds_from_beavertails_validation": thresholds,
        "probe_metrics": {
            name: binary_metrics(results, name, float(thresholds.get(name, 0.5)))
            for name in head_names
        },
        "by_scenario": {},
        "by_variant": {},
        "by_scenario_variant": {},
        "results_csv": str(output),
        "max_new_tokens": args.max_new_tokens if args.dataset == "mm_safetybench" else None,
        "batch_size": args.batch_size if args.dataset == "mm_safetybench" else None,
        "judge_enabled": bool(args.dataset == "mm_safetybench" and args.judge),
        "judge_includes_image": bool(args.dataset == "mm_safetybench" and args.judge and args.judge_with_image),
        "inference_aggregation": (
            f"max_ema_probability(alpha={ema_alpha})" if is_attention_swim else "max_rolling_mean_logit"
        ),
        "label_source": "BeaverTails-V response-safety annotations + one-vs-rest risk labels" if args.dataset == "beavertails" else (
            "multimodal LLM judge labels" if args.judge and args.judge_with_image else
            "text-only LLM judge labels" if args.judge else
            "cached LLM judge labels" if any(item.get(f"judge_label_{name}") in (0, 1) for item in results for name in head_names) else
            "none (MM-SafetyBench attack prompts only)"
        ),
    }
    if args.dataset == "mm_safetybench":
        scenario_groups: dict[str, list[dict]] = defaultdict(list)
        variant_groups: dict[str, list[dict]] = defaultdict(list)
        scenario_variant_groups: dict[str, list[dict]] = defaultdict(list)
        for item in results:
            scenario_groups[item["scenario"]].append(item)
            variant_groups[item["attack_variant"]].append(item)
            scenario_variant_groups[f"{item['scenario']}|{item['attack_variant']}"].append(item)
        summary["by_scenario"] = {
            name: {head: binary_metrics(group, head, float(thresholds.get(head, 0.5))) for head in head_names}
            for name, group in sorted(scenario_groups.items())
        }
        summary["by_variant"] = {
            name: {head: binary_metrics(group, head, float(thresholds.get(head, 0.5))) for head in head_names}
            for name, group in sorted(variant_groups.items())
        }
        summary["by_scenario_variant"] = {
            name: {head: binary_metrics(group, head, float(thresholds.get(head, 0.5))) for head in head_names}
            for name, group in sorted(scenario_variant_groups.items())
        }
        has_cached_judge_labels = any(
            item.get(f"judge_label_{name}") in (0, 1) for item in results for name in head_names
        )
        if not args.judge and not has_cached_judge_labels:
            summary["note"] = "Without --judge, MM-SafetyBench has attack-positive prompts only; AUROC/ASR cannot be computed from probe scores alone."
        elif not args.judge:
            summary["note"] = "Judge labels were reused from the input CSV; they are noisy pseudo-ground-truth, not human-verified truth."
        else:
            summary["note"] = "LLM-judge labels are noisy pseudo-ground-truth; compare them as exploratory benchmark labels, not human-verified truth. The image is included only when --judge-with-image is set."
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
