#!/usr/bin/env python3
"""Normalize downloaded BeaverTails-V into response-level JSONL manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from datasets import load_from_disk


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/beavertails_v/configs"
OUT = ROOT / "data/processed/beavertails_v"
IMAGE_ROOT = ROOT / "data/raw/beavertails_v/images"


def safe_to_label(value: Any) -> int | None:
    """Return unsafe=1 / safe=0, retaining unknown annotations as unlabeled."""
    if isinstance(value, bool):
        return int(not value)
    text = str(value or "").strip().lower()
    if text in {"safe", "yes", "true", "1"}:
        return 0
    if text in {"unsafe", "no", "false", "0"}:
        return 1
    return None


def save_image(image: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(image, "save"):
        image.convert("RGB").save(destination, format="JPEG", quality=95)
        return
    if isinstance(image, dict):
        if image.get("bytes"):
            destination.write_bytes(image["bytes"])
            return
        if image.get("path"):
            from PIL import Image

            with Image.open(image["path"]) as opened:
                opened.convert("RGB").save(destination, format="JPEG", quality=95)
            return
    raise ValueError(f"Unsupported image value: {type(image)!r}")


def image_key(row: dict[str, Any], config: str, index: int) -> str:
    image = row.get("image")
    if isinstance(image, dict) and image.get("path"):
        seed = str(image["path"])
    else:
        seed = f"{config}:{index}:{row.get('question', '')}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--image-dir", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-images", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be between 0 and 1")
    configs_dir = args.raw
    if not configs_dir.is_dir():
        raise SystemExit(f"Downloaded config directory not found: {configs_dir}\nRun scripts/download_beavertails_v.py first.")

    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "official_evaluation": []}
    source_groups: dict[str, list[dict[str, Any]]] = {"train": [], "evaluation": []}
    unknown_labels = 0
    image_count = 0

    for saved_config in sorted(path for path in configs_dir.iterdir() if path.is_dir()):
        dataset = load_from_disk(str(saved_config))
        config = saved_config.name
        for official_split, table in dataset.items():
            if official_split not in {"train", "evaluation"}:
                continue
            for index, raw in enumerate(table):
                key = image_key(raw, config, index)
                image_path = args.image_dir / f"{key}.jpg"
                if not image_path.exists() or args.overwrite_images:
                    save_image(raw.get("image"), image_path)
                    image_count += 1
                image_rel = image_path.relative_to(ROOT).as_posix() if image_path.is_relative_to(ROOT) else str(image_path)
                group_id = f"{config}:{index}"
                for response_no in (1, 2):
                    response = raw.get(f"response_{response_no}")
                    if not isinstance(response, str) or not response.strip():
                        continue
                    label = safe_to_label(raw.get(f"is_response_{response_no}_safe"))
                    if label is None:
                        unknown_labels += 1
                    record = {
                        "id": f"{group_id}:response_{response_no}",
                        "source_group_id": group_id,
                        "source_dataset": "BeaverTails-V",
                        "source_split": official_split,
                        "category": raw.get("category") or config,
                        "question": raw.get("question", ""),
                        "image_path": image_rel,
                        "response": response,
                        "response_id": response_no,
                        "label": label,
                        "is_unsafe": label,
                        "image_severity": raw.get("image_severity"),
                        "helpful_response_id": raw.get("more_helpful_response_id"),
                        "safer_response_id": raw.get("safer_response_id"),
                        "harmless_rate": raw.get(f"response_{response_no}_harmless_rate"),
                        "helpful_rate": raw.get(f"response_{response_no}_helpful_rate"),
                        "response_source": raw.get(f"response_{response_no}_from"),
                    }
                    source_groups[official_split].append(record)

    rng = random.Random(args.seed)
    by_group: dict[str, list[dict[str, Any]]] = {}
    for record in source_groups["train"]:
        by_group.setdefault(record["source_group_id"], []).append(record)
    group_ids = list(by_group)
    rng.shuffle(group_ids)
    validation_n = max(1, round(len(group_ids) * args.validation_fraction)) if group_ids else 0
    validation_groups = set(group_ids[:validation_n])
    for group, records in by_group.items():
        split_rows["validation" if group in validation_groups else "train"].extend(records)
    split_rows["official_evaluation"] = source_groups["evaluation"]

    for name, rows in split_rows.items():
        write_jsonl(args.output / f"{name}.jsonl", rows)
        print(f"{name}: {len(rows)} records / {len({r['source_group_id'] for r in rows})} source examples")
    print(f"Images materialized this run: {image_count}")
    print(f"Records with unknown safety annotation: {unknown_labels}")
    print(f"Output directory: {args.output}")


if __name__ == "__main__":
    main()
