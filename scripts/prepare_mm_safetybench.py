#!/usr/bin/env python3
"""Build an image/question manifest from an official MM-SafetyBench checkout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("SD", "SD_TYPO", "TYPO")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Path to the official MM-SafetyBench repository")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/mm_safetybench.jsonl")
    parser.add_argument("--allow-missing-images", action="store_true")
    args = parser.parse_args()
    questions_dir = args.source / "data/processed_questions"
    images_dir = args.source / "data/imgs"
    if not questions_dir.is_dir():
        raise SystemExit(f"Questions directory missing: {questions_dir}")
    if not images_dir.is_dir():
        raise SystemExit(f"Image directory missing: {images_dir}; fetch and unpack the official image archive first")

    # The official zip wraps all scenarios in a top-level directory. Keep the
    # extracted layout intact and point rows at that directory when present.
    wrapped_images_dir = images_dir / "MM-SafetyBench(imgs)"
    if wrapped_images_dir.is_dir():
        images_dir = wrapped_images_dir

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for question_file in sorted(questions_dir.glob("*.json")):
        scenario = question_file.stem
        questions = json.loads(question_file.read_text(encoding="utf-8"))
        for question_id, item in questions.items():
            for variant in VARIANTS:
                image = images_dir / scenario / variant / f"{question_id}.jpg"
                if not image.is_file():
                    missing.append(str(image))
                    continue
                rows.append({
                    "id": f"{scenario}:{question_id}:{variant}",
                    "source_dataset": "MM-SafetyBench",
                    "split": "test",
                    "scenario": scenario,
                    "question_id": str(question_id),
                    "attack_variant": variant,
                    "question": item.get("Rephrased Question") or item.get("Changed Question") or item.get("Question", ""),
                    "original_question": item.get("Question", ""),
                    "changed_question": item.get("Changed Question", ""),
                    "key_phrase": item.get("Key Phrase", ""),
                    "image_path": str(image.resolve()),
                    "label": 1,
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} image-question rows -> {args.output}")
    print(f"Missing image variants: {len(missing)}")
    if missing:
        for path in missing[:10]:
            print(f"MISSING {path}")
        if not args.allow_missing_images:
            raise SystemExit("Manifest is partial because images are missing; see official dataset download instructions.")


if __name__ == "__main__":
    main()
