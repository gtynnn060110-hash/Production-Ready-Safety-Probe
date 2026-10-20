"""Independent readouts, validation thresholds, and ID/OOD metrics/predictions."""

import logging
from dataclasses import asdict

from ..artifacts import Artifacts
from ..config import Config
from ..contracts import HEADS
from ..engine import feature_manifest, load_checkpoint, loader, predict, run_signature, setup
from ..io import SCHEMA, checked_manifest, file_hash, stage_lock, write_json, write_jsonl
from ..metrics import calibrate, evaluate_scores
from ..probes import build_probe


def save_predictions(path, data, mode):
    write_jsonl(
        path,
        (
            {
                "id": name,
                "logits": data["scores"][mode][i].tolist(),
                "labels": data["labels"][i].tolist(),
                "label_mask": data["label_mask"][i].tolist(),
                "response_tokens": data["response_lengths"][i],
            }
            for i, name in enumerate(data["ids"])
        ),
    )


def run(config: Config, paths: Artifacts) -> None:
    setup(config.seed, config.training.device)
    manifest = feature_manifest(config, paths)
    signature = run_signature(config, paths)
    with stage_lock(paths.run):
        checked_manifest(paths.run / "manifest.json", signature)
        checkpoint = paths.run / "checkpoints/best.pt"
        state = load_checkpoint(checkpoint, signature)
        model = build_probe(config.probe, input_dim=manifest["input_dim"], num_heads=len(HEADS))
        model.load_state_dict(state["model"])
        model.to(config.training.device)
        readouts = config.evaluation.readouts
        primary_fpr = (
            config.evaluation.fpr_targets[0]
            if config.evaluation.threshold_metric == "fpr"
            else None
        )
        val = predict(
            model, loader(config, paths, manifest, "validation"), config, manifest, readouts
        )
        results, thresholds = {}, {}
        for mode in readouts:
            thresholds[mode] = calibrate(
                val["labels"], val["label_mask"], val["scores"][mode], config.evaluation.fpr_targets
            )
            results[mode] = {
                "schema": SCHEMA,
                "heads": list(HEADS),
                "readout": mode,
                "evaluation_config": asdict(config.evaluation),
                "checkpoint_sha256": file_hash(checkpoint),
                "feature_manifest_sha256": signature["feature_manifest_hash"],
                "checkpoint_epoch": state["epoch"] + 1,
                "threshold_source": "BeaverTails validation only; thresholds in logits",
                "primary_threshold_rule": {
                    "metric": config.evaluation.threshold_metric,
                    "fpr_target": primary_fpr,
                },
                "thresholds": thresholds[mode],
                "splits": {},
            }
        for split in ["validation", *config.evaluation.targets]:
            data = (
                val
                if split == "validation"
                else predict(
                    model, loader(config, paths, manifest, split), config, manifest, readouts
                )
            )
            for mode in readouts:
                results[mode]["splits"][split] = evaluate_scores(
                    data["labels"], data["label_mask"], data["scores"][mode], thresholds[mode],
                    threshold_metric=config.evaluation.threshold_metric, fpr_target=primary_fpr,
                )
                save_predictions(
                    paths.run / "evaluation" / "predictions" / f"{split}-{mode}.jsonl", data, mode
                )
        for mode, result in results.items():
            write_json(paths.run / "evaluation" / f"{mode}.json", result)
            logging.info(
                "%s Overall: %s",
                mode,
                {
                    split: {
                        key: metrics["overall"][key]
                        for key in ("n", "auroc", "auprc", "f1", "recall", "fpr")
                    }
                    for split, metrics in result["splits"].items()
                    if split != "validation"
                },
            )
