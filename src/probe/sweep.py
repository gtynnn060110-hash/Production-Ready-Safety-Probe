"""Validation-only risk-window/LR search with fixed full-history causal MQA.

Run from the repository root: python -m probe.sweep --help
Training uses the same optimizer, schedule, microbatches and epoch selection as
the ordinary runner. Only the data loader changes: a deterministic subset and
the complete validation split remain resident across trials. Test features are
never loaded or scored. Each trial can resume from its epoch checkpoint.
"""

import argparse
import copy
import itertools
import logging
import os
import shutil
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from .artifacts import Artifacts
from .config import load_config
from .contracts import HEADS
from .engine import feature_manifest, load_checkpoint, predict, run_signature, setup
from .features import FeatureDataset, collate_features
from .io import checked_manifest, file_hash, fingerprint, read_json, stage_lock, write_json
from .losses import build_loss
from .metrics import calibrate, evaluate_scores
from .probes import build_probe
from .stages.train import train_epochs


def write_config(path, config):
    # PyYAML's YAML 1.1 resolver treats JSON exponents such as 1e-06 as strings.
    # Export actual YAML so the selected configuration can be loaded unchanged.
    temp = path.with_suffix(".yaml.tmp")
    temp.write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")
    temp.replace(path)


class ResidentFeatures(Dataset):
    """Packed raw features; preserves the normal loader's normalization and order."""

    def __init__(self, root, manifest, split, *, limit=None, seed=42, device="cpu"):
        if split not in {"train", "validation"}:
            raise ValueError("Hyperparameter search only permits train/validation features")
        source = FeatureDataset(root, manifest, split)
        indices = list(range(len(source)))
        if limit is not None:
            if type(limit) is not int or limit < 1:
                raise ValueError("Training subset size must be a positive integer")
            indices = sorted(
                indices, key=lambda i: fingerprint([seed, split, source.rows[i]["id"]])
            )[:limit]
        indices.sort(key=lambda i: source.rows[i]["id"])
        total = sum(source.rows[i]["stop"] - source.rows[i]["start"] for i in indices)
        first = source[indices[0]]
        packed = torch.empty((total, manifest["input_dim"]), dtype=first["hidden_states"].dtype)
        masks = torch.empty(total, dtype=torch.bool)
        self.rows = []
        cursor = 0
        for index in indices:
            row = source[index]
            stop = cursor + len(row["hidden_states"])
            packed[cursor:stop].copy_(row.pop("hidden_states"))
            masks[cursor:stop].copy_(row.pop("response_mask"))
            self.rows.append(row | {"start": cursor, "stop": stop})
            cursor = stop
        self.hidden, self.mask = packed.to(device), masks.to(device)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        start, stop = row["start"], row["stop"]
        return {
            "id": row["id"],
            "hidden_states": self.hidden[start:stop],
            "response_mask": self.mask[start:stop],
            "labels": row["labels"],
            "label_mask": row["label_mask"],
        }


def resident_loader(datasets):
    def make_loader(config, paths, manifest, split, train=False, epoch=0):
        if split not in {"train", "validation"}:
            raise ValueError("Test data cannot be used during hyperparameter search")
        dataset = datasets[split]
        batch_size = config.training.micro_batch_size if train else config.evaluation.batch_size
        if train and len(dataset) < batch_size:
            raise ValueError("Training subset is smaller than one microbatch")
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            drop_last=train,
            collate_fn=collate_features,
            num_workers=0,
            generator=torch.Generator().manual_seed(config.seed + epoch),
        )

    return make_loader


def trial_configs(base, risk_windows, learning_rates):
    for risk, lr in itertools.product(risk_windows, learning_rates):
        config = copy.deepcopy(base)
        config.probe.params.update(context_window=0, risk_window=risk)
        config.training.learning_rate = lr
        config.name = f"mqa0_risk{risk}_lr{lr:g}_seed{config.seed}"
        config.validate()
        # Constructor validation catches invalid component settings before a long search.
        build_probe(config.probe, input_dim=8, num_heads=len(HEADS))
        yield config


def run_trial(config, paths, manifest, make_loader, subset_signature):
    signature = run_signature(config, paths) | {"sweep_subset": subset_signature}
    with stage_lock(paths.run):
        result_path = paths.run / "result.json"
        if result_path.exists():
            return checked_manifest(result_path, signature)["result"]
        setup(config.seed, config.training.device)
        model = build_probe(config.probe, input_dim=manifest["input_dim"], num_heads=len(HEADS))
        parameters = sum(p.numel() for p in model.parameters())
        model.to(config.training.device)
        write_config(paths.run / "config.yaml", config)
        train_epochs(
            config, paths, manifest, model, build_loss(config.loss), signature,
            loader_factory=make_loader,
        )
        state = load_checkpoint(paths.run / "checkpoints/best.pt", signature)
        model.load_state_dict(state["model"])
        modes = ["max", "mean", "last"]
        validation = predict(
            model, make_loader(config, paths, manifest, "validation"), config, manifest, modes
        )
        metrics = {}
        for mode in modes:
            thresholds = calibrate(
                validation["labels"], validation["label_mask"], validation["scores"][mode],
                config.evaluation.fpr_targets,
            )
            metrics[mode] = evaluate_scores(
                validation["labels"], validation["label_mask"], validation["scores"][mode],
                thresholds,
            )
        result = {
            "name": config.name,
            "context_window": config.probe.params["context_window"],
            "risk_window": config.probe.params["risk_window"],
            "learning_rate": config.training.learning_rate,
            "parameters": parameters,
            "best_epoch": state["epoch"] + 1,
            "selection_metric": "validation_overall_max_auprc",
            "selection_score": metrics["max"]["overall"]["auprc"],
            "validation": metrics,
            "run_dir": str(paths.run),
        }
        write_json(result_path, {
            "schema": 1,
            "signature": signature,
            "result": result,
            "files": {
                name: file_hash(paths.run / name)
                for name in ("checkpoints/best.pt", "checkpoints/last.pt", "history.json")
            },
        })
        return result


def now():
    return datetime.now(UTC).isoformat()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append")
    parser.add_argument("--set", dest="overrides", action="append")
    parser.add_argument("--risk-windows", type=int, nargs="+", default=[10, 20, 40])
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[5e-5, 1e-4, 3e-4])
    parser.add_argument("--train-samples", type=int, default=8192)
    parser.add_argument("--cache-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/sweeps/mqa_full_rmattn_weighted")
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(4)
    config = load_config(args.config or [
        Path("configs/base.yaml"), Path("configs/probes/mqa_rmattn.yaml"),
        Path("configs/sweeps/mqa_rmattn_weighted.yaml"),
    ], args.overrides)
    if config.probe.name != "mqa_rmattn":
        raise ValueError("This search is for the mqa_rmattn probe")
    config.probe.params["context_window"] = 0
    if args.train_samples < config.training.micro_batch_size:
        raise ValueError("Training subset must contain at least one microbatch")
    if config.training.selection_readout != "max":
        raise ValueError("This search selects checkpoints and trials using the max readout")
    candidates = list(trial_configs(config, args.risk_windows, args.learning_rates))
    if len({c.name for c in candidates}) != len(candidates):
        raise ValueError("Search axes must not contain duplicate candidates")
    paths = Artifacts.from_config(config)
    job = args.output.resolve()
    with stage_lock(job):
        code_root = Path(__file__).parent
        plan = {
            "config": asdict(config),
            "trials": [asdict(c) for c in candidates],
            "train_samples": args.train_samples,
            "subset_selection": "lowest hash(split_seed, train, sample_id), then ID order",
            "validation": "complete cached validation split",
            "test_data_used": False,
            "cache_device": args.cache_device,
            "feature_manifest_sha256": file_hash(paths.features / "manifest.json"),
            "source_sha256": {
                str(p.relative_to(code_root)): file_hash(p)
                for p in sorted(code_root.rglob("*.py"))
            },
        }
        plan_path = job / "plan.json"
        if plan_path.exists() and read_json(plan_path) != plan:
            raise ValueError("Search inputs/source changed; choose a new --output directory")
        write_json(plan_path, plan)
        shutil.copytree(code_root, job / "snapshot/probe", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__"))
        started = time.monotonic()
        status = {"state": "verifying_cache", "pid": os.getpid(), "started_utc": now(),
                  "total_trials": len(candidates), "completed_trials": 0}
        write_json(job / "status.json", status)
        try:
            manifest = feature_manifest(config, paths, splits=("train", "validation"))
            status["state"] = "loading_features"
            write_json(job / "status.json", status)
            device = config.training.device if args.cache_device == "cuda" else "cpu"
            datasets = {
                split: ResidentFeatures(
                    paths.features, manifest, split, seed=config.data.split_seed, device=device,
                    limit=args.train_samples if split == "train" else None,
                )
                for split in ("train", "validation")
            }
            members = {s: [r["id"] for r in d.rows] for s, d in datasets.items()}
            write_json(job / "members.json", members)
            subset = {s: {"samples": len(ids), "ids_sha256": fingerprint(ids)}
                      for s, ids in members.items()}
            make_loader = resident_loader(datasets)
            results = []
            for i, candidate in enumerate(candidates, 1):
                status.update(state="running", current_trial=candidate.name,
                              completed_trials=i - 1, trial_started_utc=now())
                write_json(job / "status.json", status)
                trial_started = time.monotonic()
                trial_paths = Artifacts(
                    paths.prepared, paths.features, job / "trials" / candidate.name
                )
                result = run_trial(candidate, trial_paths, manifest, make_loader, subset)
                results.append(result)
                ranked = sorted(results, key=lambda r: (-r["selection_score"], r["name"]))
                write_json(job / "summary.json", {
                    "protocol": plan | {"membership": subset}, "results": ranked,
                    "best": ranked[0], "complete": i == len(candidates),
                })
                logging.info("TRIAL %s/%s %s val AP=%.6f elapsed=%.1fs", i, len(candidates),
                             candidate.name, result["selection_score"],
                             time.monotonic() - trial_started)
                status.update(completed_trials=i, best=ranked[0]["name"],
                              best_validation_ap=ranked[0]["selection_score"])
                write_json(job / "status.json", status)
            best = next(c for c in candidates if c.name == ranked[0]["name"])
            best = copy.deepcopy(best)
            best.name = "selected_mqa_rmattn_weighted"
            write_config(job / "best_config.yaml", best)
            status.update(state="complete", current_trial=None, completed_utc=now(),
                          elapsed_seconds=time.monotonic() - started)
            write_json(job / "status.json", status)
        except BaseException as exc:
            status.update(state="failed", error=repr(exc), updated_utc=now())
            write_json(job / "status.json", status)
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
