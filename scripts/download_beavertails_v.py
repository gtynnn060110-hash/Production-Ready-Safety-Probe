#!/usr/bin/env python3
"""Download BeaverTails-V into the project-local raw data directory."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import get_dataset_config_names, load_dataset


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "saferlhf-v/BeaverTails-V"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw/beavertails_v")
    parser.add_argument("--config", action="append", help="Optional config/category; repeat to select several")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent category downloads (default: 4)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    args.output.mkdir(parents=True, exist_ok=True)
    configs = args.config or get_dataset_config_names(DATASET_ID)
    print(f"Dataset: {DATASET_ID}")
    print(f"Configs: {len(configs)}")
    print(f"Output: {args.output}")
    def fetch(config: str) -> tuple[str, dict[str, int]]:
        dataset = load_dataset(DATASET_ID, config, cache_dir=str(args.output / "hf_cache"))
        target = args.output / "configs" / config
        dataset.save_to_disk(str(target))
        counts = {split: len(rows) for split, rows in dataset.items()}
        return config, counts

    failed: list[tuple[str, Exception]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, config): config for config in configs}
        for future in as_completed(futures):
            config = futures[future]
            try:
                _, counts = future.result()
                print(f"Saved {config}: {counts}", flush=True)
            except Exception as error:
                failed.append((config, error))
                print(f"FAILED {config}: {type(error).__name__}: {error}", flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} config(s) failed; rerun this command to resume them.")


if __name__ == "__main__":
    main()
