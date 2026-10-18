"""Stage dispatch and side-effect-free execution plans."""

from dataclasses import asdict
from importlib import import_module

from .artifacts import Artifacts
from .config import Config
from .contracts import HEADS

STAGES = ("prepare", "extract", "train", "evaluate")


def plan(stage: str, config: Config) -> dict:
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    paths = Artifacts.from_config(config)
    prepared = paths.prepared / "manifest.json"
    features = paths.features / "manifest.json"
    checkpoint = paths.run / "checkpoints/best.pt"
    inputs, outputs = {
        "prepare": ([], [prepared]),
        "extract": ([prepared], [features]),
        "train": ([features], [checkpoint]),
        "evaluate": (
            [prepared, features, checkpoint],
            [
                paths.run / "evaluation" / f"{readout}.json"
                for readout in config.evaluation.readouts
            ],
        ),
    }[stage]
    return {
        "stage": stage,
        "status": "planned",
        "heads": list(HEADS),
        "inputs": [{"path": str(p), "exists": p.exists()} for p in inputs],
        "outputs": [str(p) for p in outputs],
        "config": asdict(config),
    }


def run(stage: str, config: Config) -> None:
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    module = import_module(f"probe.stages.{stage}")
    module.run(config, Artifacts.from_config(config))
