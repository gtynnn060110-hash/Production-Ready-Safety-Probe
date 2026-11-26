"""Prepare grouped train/validation/test data and benchmark labels."""

import logging
from dataclasses import asdict

from ..artifacts import DATA_VERSION, Artifacts
from ..config import Config
from ..contracts import HEADS
from ..data import CATEGORY_HEAD, EXCLUDED, label_stats, prepare_records
from ..io import SCHEMA, checked_manifest, file_hash, stage_lock, write_json, write_jsonl


def run(config: Config, paths: Artifacts) -> None:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    signature = {
        "data": asdict(config.data),
        "heads": list(HEADS),
        "mapping": CATEGORY_HEAD,
        "excluded": sorted(EXCLUDED),
        "aggregation": "per_head_majority_ties_safe_then_filter",
        "implementation": DATA_VERSION,
    }
    with stage_lock(paths.prepared):
        target = paths.prepared / "manifest.json"
        if target.exists():
            checked_manifest(target, signature)
            logging.info("Reusing prepared data: %s", paths.prepared)
            return
        revisions = {}
        for name, repo, revision in [
            ("train", config.data.train_repo, config.data.train_revision),
            ("benchmark", config.data.benchmark_repo, config.data.benchmark_revision),
        ]:
            if name == "benchmark" and not config.data.benchmark_enabled:
                continue
            revisions[name] = revision or HfApi().dataset_info(repo).sha
        # Load benchmark first: access failures never leave an apparently complete prepare.
        benchmark = []
        if config.data.benchmark_enabled:
            benchmark = load_dataset(
                config.data.benchmark_repo,
                config.data.benchmark_config,
                split="test",
                revision=revisions["benchmark"],
            )
        sources = [
            (split, load_dataset(config.data.train_repo, split=split, revision=revisions["train"]))
            for split in config.data.source_splits
        ]
        splits, counts = prepare_records(sources, benchmark, config.data)
        full_splits = counts.pop("full_splits")
        audit = counts.pop("annotation_audit")
        if any(not splits[name] for name in ("train", "validation", "beavertails_test")):
            raise ValueError("Empty full grouped split; inspect source data and split_ratios")
        files = {}
        for name, rows in splits.items():
            path = paths.prepared / f"{name}.jsonl"
            write_jsonl(path, rows)
            files[path.name] = file_hash(path)
        audit_path = paths.prepared / "annotations.jsonl"
        write_jsonl(audit_path, audit)
        files[audit_path.name] = file_hash(audit_path)
        manifest = {
            "schema": SCHEMA,
            "signature": signature,
            "revisions": revisions,
            "files": files,
            "splits": {k: label_stats(v) for k, v in splits.items()},
            "full_splits": full_splits,
            "counts": counts,
            "split_protocol": "full prompt-grouped split, then per-split subsets; custom test",
        }
        write_json(target, manifest)
        logging.info("Prepared %s", {k: len(v) for k, v in splits.items()})
