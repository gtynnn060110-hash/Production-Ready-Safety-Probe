#!/usr/bin/env python3
"""Apply the multimodal LLM judge to responses already saved by eval_rolling_probe.py."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.probe_runtime import judge_answer, read_jsonl  # noqa: E402

HEADS = ("physical_harm", "hate_speech", "privacy", "overall_safety")


def label_value(value: object) -> int | None:
    if value in (0, 1):
        return int(value)
    text = str(value).strip().lower()
    if text in {"0", "1"}:
        return int(text)
    return None


def probe_metrics(rows: list[dict], head: str) -> dict:
    pairs = []
    for row in rows:
        label = label_value(row.get(f"judge_label_{head}"))
        probability = row.get(f"probe_probability_{head}")
        if label is not None and probability not in (None, ""):
            pairs.append((label, float(probability), float(row.get(f"threshold_{head}") or 0.5)))
    if not pairs:
        return {"n": len(rows), "labeled_n": 0}
    labels = [item[0] for item in pairs]
    scores = [item[1] for item in pairs]
    predictions = [int(score >= threshold) for _, score, threshold in pairs]
    result = {
        "n": len(rows),
        "labeled_n": len(pairs),
        "unsafe_rate_reference": sum(labels) / len(labels),
        "probe_recall_at_val_threshold": sum(p and y for p, y in zip(predictions, labels)) / max(1, sum(labels)),
        "probe_f1_at_val_threshold": float(f1_score(labels, predictions, zero_division=0)),
        "probe_accuracy_at_val_threshold": float(accuracy_score(labels, predictions)),
        "probe_auroc_vs_reference": None,
        "probe_auprc_vs_reference": None,
    }
    if len(set(labels)) == 2:
        result["probe_auroc_vs_reference"] = float(roc_auc_score(labels, scores))
        result["probe_auprc_vs_reference"] = float(average_precision_score(labels, scores))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Generation-stage CSV from eval_rolling_probe.py (without --judge)")
    parser.add_argument("--output", type=Path, default=None, help="Judged CSV; defaults to <input-stem>_judged.csv")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/processed/mm_safetybench.jsonl")
    parser.add_argument("--judge-with-image", action="store_true", help="Include each MM-SafetyBench image in the judge request")
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted judge pass using the partial output CSV")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent judge requests (default: 8)")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if not args.input.is_file():
        raise SystemExit(f"Input CSV not found: {args.input}")
    if not os.environ.get("OPENAI_MODEL"):
        raise SystemExit("Set OPENAI_MODEL (and usually OPENAI_API_KEY) before judging.")
    output = args.output or args.input.with_name(f"{args.input.stem}_judged.csv")
    if output.resolve() == args.input.resolve():
        raise SystemExit("Input and output must be different files.")
    manifest_path = output.with_suffix(".manifest.json")
    manifest = {
        "input": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "dataset": str(args.dataset.resolve()),
        "include_image": args.judge_with_image,
        "judge_model": os.environ.get("OPENAI_MODEL"),
        "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.minimax.io/v1").rstrip("/"),
    }

    with args.input.open(newline="", encoding="utf-8") as handle:
        input_rows = list(csv.DictReader(handle))
    if not input_rows:
        raise SystemExit(f"No rows in input CSV: {args.input}")
    input_by_id = {str(row.get("id", "")): row for row in input_rows}
    if "" in input_by_id or len(input_by_id) != len(input_rows):
        raise SystemExit("Input CSV must have non-empty, unique id values.")
    for row in input_rows:
        if not row.get("response"):
            raise SystemExit(f"Missing generated response for row {row.get('id')!r}.")

    source_rows = read_jsonl(args.dataset)
    source_by_id = {str(row.get("id", "")): row for row in source_rows}
    missing = set(input_by_id) - set(source_by_id)
    if missing:
        raise SystemExit(f"Input rows are missing from the MM-SafetyBench source JSONL (example: {next(iter(missing))}).")

    cached: dict[str, dict] = {}
    if args.resume:
        if not output.is_file() or not manifest_path.is_file():
            raise SystemExit("Cannot resume: expected both the partial judged CSV and its .manifest.json file.")
        saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved_manifest != manifest:
            raise SystemExit("Judge resume settings do not match the original run; use the original input/model/image options.")
        with output.open(newline="", encoding="utf-8") as handle:
            cached_rows = list(csv.DictReader(handle))
        for row in cached_rows:
            row_id = str(row.get("id", ""))
            if row_id not in input_by_id or row_id in cached:
                raise SystemExit(f"Partial judged CSV contains an unexpected or duplicate row id: {row_id!r}")
            cached[row_id] = row
    else:
        if output.exists():
            raise SystemExit(f"Output already exists: {output}; choose another --output or pass --resume.")
        output.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fieldnames = list(input_rows[0].keys())
    for key in ("judge_text", *(f"judge_label_{head}" for head in HEADS)):
        if key not in fieldnames:
            fieldnames.append(key)
    output.parent.mkdir(parents=True, exist_ok=True)
    todo_rows = [
        row for row in input_rows
        if not (
            cached.get(str(row["id"]))
            and cached[str(row["id"])].get("judge_text")
            and all(label_value(cached[str(row["id"])].get(f"judge_label_{head}")) in (0, 1) for head in HEADS)
        )
    ]
    mode = "a" if args.resume else "w"
    judged_now = 0
    failed_ids: list[str] = []
    with output.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not args.resume:
            writer.writeheader()
            handle.flush()

        # API calls run concurrently, but only the main thread appends results
        # to CSV, keeping checkpoints valid for --resume.
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    judge_answer,
                    source_by_id[str(row["id"])],
                    row["response"],
                    include_image=args.judge_with_image,
                ): row
                for row in todo_rows
            }
            with tqdm(total=len(todo_rows), desc="judging", unit="row") as progress:
                for future in as_completed(futures):
                    row = futures[future]
                    row_id = str(row["id"])
                    try:
                        labels, judge_text = future.result()
                    except Exception as exc:
                        failed_ids.append(row_id)
                        progress.set_postfix(done=judged_now, failed=len(failed_ids))
                        progress.update(1)
                        print(f"\njudge failed for {row_id}: {exc}", flush=True)
                        continue
                    judged_row = dict(row)
                    judged_row["judge_text"] = judge_text
                    for head in HEADS:
                        judged_row[f"judge_label_{head}"] = labels[head]
                    writer.writerow(judged_row)
                    handle.flush()
                    cached[row_id] = judged_row
                    judged_now += 1
                    progress.set_postfix(done=judged_now, failed=len(failed_ids))
                    progress.update(1)

    final_rows = [cached.get(str(row["id"]), row) for row in input_rows]
    head_summary = {}
    for head in HEADS:
        labels = [label_value(row.get(f"judge_label_{head}")) for row in final_rows]
        labels = [value for value in labels if value is not None]
        head_summary[head] = {
            "labeled_n": len(labels),
            "positive_n": sum(labels),
            "positive_rate": sum(labels) / len(labels) if labels else None,
        }
    summary = {
        "input_csv": str(args.input),
        "results_csv": str(output),
        "dataset": str(args.dataset),
        "rows": len(final_rows),
        "judge_enabled": True,
        "judge_model": os.environ["OPENAI_MODEL"],
        "judge_includes_image": args.judge_with_image,
        "newly_judged_this_run": judged_now,
        "failed_this_run": len(failed_ids),
        "failed_ids_this_run": failed_ids,
        "workers": args.workers,
        "labels": head_summary,
        "probe_metrics_vs_judge": {head: probe_metrics(final_rows, head) for head in HEADS},
        "note": "LLM-judge labels are noisy pseudo-ground-truth, not human-verified annotations.",
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
