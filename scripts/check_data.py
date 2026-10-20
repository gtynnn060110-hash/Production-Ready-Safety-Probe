#!/usr/bin/env python3
"""Validate project manifests and referenced image files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def inspect(path: Path, required: set[str], image_root: Path | None, min_rows: int = 1) -> bool:
    if not path.is_file():
        print(f"NOT READY {path}")
        return False
    n = 0
    labels: Counter[str] = Counter()
    missing: list[str] = []
    groups: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            absent = required - row.keys()
            if absent:
                raise ValueError(f"{path}:{line_no} missing keys: {sorted(absent)}")
            n += 1
            labels[str(row.get("label"))] += 1
            if row.get("source_group_id"):
                groups.add(row["source_group_id"])
            if image_root and row.get("image_path"):
                image = Path(row["image_path"])
                if not image.is_absolute():
                    image = ROOT / image
                if not image.is_file():
                    missing.append(str(image))
    ready = n >= min_rows and not missing
    status = "OK" if ready else "NOT READY"
    suffix = f", expected_at_least={min_rows}" if n < min_rows else ""
    print(f"{status} {path}: rows={n}, groups={len(groups) or 'n/a'}, labels={dict(labels)}, missing_images={len(missing)}{suffix}")
    for item in missing[:5]:
        print(f"MISSING {item}")
    return ready


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--allow-mm-subset", action="store_true", help="Accept a non-empty MM-SafetyBench subset instead of requiring the full 5,040 pairs")
    args = parser.parse_args()
    data = args.root / "data/processed"
    bt = data / "beavertails_v"
    all_ready = True
    for split in ("train", "validation", "official_evaluation"):
        all_ready &= inspect(bt / f"{split}.jsonl", {"id", "source_group_id", "question", "response", "image_path", "label"}, args.root)
    all_ready &= inspect(
        data / "mm_safetybench.jsonl",
        {"id", "scenario", "attack_variant", "question", "image_path", "label"},
        args.root,
        min_rows=1 if args.allow_mm_subset else 5040,
    )
    if not all_ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
