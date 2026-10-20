"""Typed configuration, small YAML overlays, and explicit command-line overrides."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass
class DataConfig:
    train_repo: str = "PKU-Alignment/BeaverTails"
    train_revision: str | None = None
    source_splits: list[str] = field(default_factory=lambda: ["330k_train", "330k_test"])
    benchmark_repo: str = "allenai/wildguardmix"
    benchmark_config: str = "wildguardtest"
    benchmark_revision: str | None = None
    split_seed: int = 42
    split_ratios: list[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])
    group_by: str = "prompt"
    label_policy: str = "beavertails_four_heads_v1"
    benchmark_enabled: bool = True
    max_samples: int | dict[str, int | None] | None = None  # hard caps AFTER full splitting


@dataclass
class FeatureConfig:
    model: str = "Qwen/Qwen3.5-9B"
    revision: str | None = None
    block_indices: list[int] = field(default_factory=lambda: [19])
    tap: str = "block_output_pre_final_norm"
    normalize: str = "rms_no_affine"
    max_length: int = 4096
    overflow: str = "drop_training_preserve_test"
    dtype: str = "bfloat16"
    batch_size: int = 1
    cache_prompt: bool = True
    norm_eps: float = 1e-6
    shard_tokens: int = 32768
    max_cache_gb: float = 500.0  # explicit disk budget; estimate checked before model loading
    device: str = "cuda"
    splits: list[str] | None = None  # None: all prepared splits; pilot: train/validation only


@dataclass
class ComponentConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    epochs: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    dtype: str = "bfloat16"
    selection_metric: str = "overall_auprc"
    selection_readout: str = "max"
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    device: str = "cuda"
    num_workers: int = 0
    log_every: int = 50
    resume: bool = True


@dataclass
class EvaluationConfig:
    readouts: list[str] = field(default_factory=lambda: ["last", "mean", "max"])
    targets: list[str] = field(default_factory=lambda: ["beavertails_test", "wildguardtest"])
    threshold_metric: str = "f1"
    fpr_targets: list[float] = field(default_factory=lambda: [0.01, 0.05])
    batch_size: int = 8


@dataclass
class Config:
    name: str = "baseline"
    seed: int = 42
    output_dir: str = "artifacts"
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    probe: ComponentConfig = field(default_factory=lambda: ComponentConfig("linear"))
    loss: ComponentConfig = field(default_factory=lambda: ComponentConfig("bce"))
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        if not isinstance(self.name, str) or not self.name or Path(self.name).name != self.name:
            raise ValueError("name must be a non-empty filename without directory components")
        if self.name in {".", ".."}:
            raise ValueError("name cannot be '.' or '..'")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("output_dir must be a non-empty path")
        ratios = self.data.split_ratios
        if type(self.data.split_seed) is not int or self.data.split_seed < 0:
            raise ValueError("data.split_seed must be a non-negative integer")
        if (
            not isinstance(ratios, list)
            or len(ratios) != 3
            or any(type(r) not in (int, float) or not 0 < r < 1 for r in ratios)
            or not math.isclose(sum(ratios), 1.0)
        ):
            raise ValueError("data.split_ratios must contain three positive ratios summing to 1")
        layers = self.features.block_indices
        if (
            not isinstance(layers, list)
            or not layers
            or any(type(i) is not int or i < 0 for i in layers)
            or len(set(layers)) != len(layers)
        ):
            raise ValueError("features.block_indices must contain distinct non-negative integers")
        for key, component in (("probe", self.probe), ("loss", self.loss)):
            if not isinstance(component.name, str) or not component.name:
                raise ValueError(f"{key}.name must be a non-empty string")
            if not isinstance(component.params, dict):
                raise ValueError(f"{key}.params must be a mapping")
        for key, value in (
            ("features.max_length", self.features.max_length),
            ("features.batch_size", self.features.batch_size),
            ("training.micro_batch_size", self.training.micro_batch_size),
            ("training.gradient_accumulation_steps", self.training.gradient_accumulation_steps),
            ("training.epochs", self.training.epochs),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key, value in (
            ("training.learning_rate", self.training.learning_rate),
            ("training.max_grad_norm", self.training.max_grad_norm),
        ):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be a finite positive number")
        if (
            type(self.training.weight_decay) not in (int, float)
            or not math.isfinite(self.training.weight_decay)
            or self.training.weight_decay < 0
        ):
            raise ValueError("training.weight_decay must be a finite non-negative number")
        if (
            type(self.training.warmup_ratio) not in (int, float)
            or not 0 <= self.training.warmup_ratio < 1
        ):
            raise ValueError("training.warmup_ratio must be in [0, 1)")
        readouts = self.evaluation.readouts
        if (
            not isinstance(readouts, list)
            or not readouts
            or any(r not in ("last", "mean", "max") for r in readouts)
            or len(set(readouts)) != len(readouts)
        ):
            raise ValueError("evaluation.readouts must be a non-empty subset of [last, mean, max]")
        if self.probe.name not in {"linear", "singprobe", "gemini", "mqa_rmattn"}:
            raise ValueError(f"Unknown probe: {self.probe.name}")
        if self.loss.name not in {"bce", "singprobe_weighted", "plugguard_atc"}:
            raise ValueError(f"Unknown loss: {self.loss.name}")
        if self.data.group_by != "prompt" or self.data.label_policy != "beavertails_four_heads_v1":
            raise ValueError("Only prompt grouping and beavertails_four_heads_v1 are implemented")
        limits = self.data.max_samples
        if isinstance(limits, dict):
            if limits.keys() - {"train", "validation", "beavertails_test", "wildguardtest"}:
                raise ValueError("Unknown split in data.max_samples")
            values = limits.values()
        else:
            values = [limits]
        if any(value is not None and (type(value) is not int or value < 1) for value in values):
            raise ValueError("data.max_samples must be null, a positive integer, or per-split caps")
        if (
            not isinstance(self.data.source_splits, list)
            or not self.data.source_splits
            or any(s not in {"330k_train", "330k_test"} for s in self.data.source_splits)
            or len(set(self.data.source_splits)) != len(self.data.source_splits)
        ):
            raise ValueError("Use distinct original 330k source splits; do not mix overlapping 30k")
        for value in (
            self.data.benchmark_enabled,
            self.features.cache_prompt,
            self.training.resume,
        ):
            if type(value) is not bool:
                raise ValueError("benchmark_enabled, cache_prompt, and resume must be booleans")
        for value in (self.features.device, self.training.device):
            if not isinstance(value, str) or not (
                value == "cpu"
                or value == "cuda"
                or (value.startswith("cuda:") and value.removeprefix("cuda:").isdigit())
            ):
                raise ValueError("device must be cpu, cuda, or cuda:<index>")
        for key, value in (
            ("features.shard_tokens", self.features.shard_tokens),
            ("training.log_every", self.training.log_every),
            ("evaluation.batch_size", self.evaluation.batch_size),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.training.num_workers) is not int or self.training.num_workers < 0:
            raise ValueError("training.num_workers must be non-negative")
        if self.features.tap != "block_output_pre_final_norm" or not self.features.cache_prompt:
            raise ValueError("Extraction requires raw block outputs and full prompt context")
        if self.features.normalize not in {"none", "rms_no_affine"}:
            raise ValueError("features.normalize must be none or rms_no_affine")
        selected = self.features.splits
        if selected is not None and (
            not isinstance(selected, list)
            or not selected
            or any(
                s not in ("train", "validation", "beavertails_test", "wildguardtest")
                for s in selected
            )
            or len(set(selected)) != len(selected)
        ):
            raise ValueError(
                "features.splits must be null or a non-empty list of distinct split names"
            )
        if self.features.overflow != "drop_training_preserve_test":
            raise ValueError("Only drop_training_preserve_test overflow policy is implemented")
        for value in (self.features.norm_eps, self.features.max_cache_gb):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("Feature normalization epsilon and disk budget must be positive")
        for value in (self.features.dtype, self.training.dtype):
            if value not in {"float32", "bfloat16"}:
                raise ValueError("dtype must be float32 or bfloat16")
        if self.training.selection_metric != "overall_auprc":
            raise ValueError("Only overall_auprc checkpoint selection is implemented")
        if self.training.selection_readout not in {"last", "mean", "max"}:
            raise ValueError("Invalid training.selection_readout")
        if self.evaluation.threshold_metric not in {"f1", "fpr"}:
            raise ValueError("evaluation.threshold_metric must be f1 or fpr")
        if (
            not isinstance(self.training.betas, list)
            or len(self.training.betas) != 2
            or any(type(b) not in (int, float) or not 0 <= b < 1 for b in self.training.betas)
        ):
            raise ValueError("training.betas must contain two values in [0, 1)")
        if (
            not isinstance(self.evaluation.targets, list)
            or not self.evaluation.targets
            or any(t not in {"beavertails_test", "wildguardtest"} for t in self.evaluation.targets)
        ):
            raise ValueError("Invalid evaluation.targets")
        if not self.data.benchmark_enabled and "wildguardtest" in self.evaluation.targets:
            raise ValueError("WildGuardTest target requires data.benchmark_enabled=true")
        if not isinstance(self.evaluation.fpr_targets, list) or any(
            type(f) not in (int, float) or not 0 < f < 1 for f in self.evaluation.fpr_targets
        ):
            raise ValueError("evaluation.fpr_targets must be between 0 and 1")
        if self.evaluation.threshold_metric == "fpr" and len(self.evaluation.fpr_targets) != 1:
            raise ValueError(
                "FPR threshold selection requires exactly one evaluation.fpr_targets value"
            )


T = TypeVar("T")


def _section(cls: type[T], values: dict[str, Any], name: str) -> T:
    if not isinstance(values, dict):
        raise ValueError(f"{name} must be a mapping")
    unknown = values.keys() - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown keys in {name}: {', '.join(sorted(unknown))}")
    try:
        return cls(**values)
    except TypeError as exc:
        raise ValueError(f"Invalid {name}: {exc}") from exc


def _merge(base: dict, overlay: dict) -> None:
    for key, value in overlay.items():
        # Components are replaced as units: never carry parameters across architectures.
        if (
            key not in {"probe", "loss"}
            and isinstance(value, dict)
            and isinstance(base.get(key), dict)
        ):
            _merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def load_config(paths: list[Path], overrides: list[str] | None = None) -> Config:
    values = asdict(Config())
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            overlay = yaml.safe_load(handle)
        if not isinstance(overlay, dict):
            raise ValueError(f"Config must contain a mapping: {path}")
        _merge(values, overlay)
    for override in overrides or []:
        key, separator, value = override.partition("=")
        if not separator or not key or any(not part for part in key.split(".")):
            raise ValueError(f"Expected --set key=value, got: {override}")
        node = values
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise ValueError(f"Unknown config path: {key}")
            node = node[part]
        if parts[-1] not in node and parts[:-1] not in (["probe", "params"], ["loss", "params"]):
            raise ValueError(f"Unknown config path: {key}")
        node[parts[-1]] = yaml.safe_load(value)
    for key, cls in (
        ("data", DataConfig),
        ("features", FeatureConfig),
        ("probe", ComponentConfig),
        ("loss", ComponentConfig),
        ("training", TrainingConfig),
        ("evaluation", EvaluationConfig),
    ):
        values[key] = _section(cls, values[key], key)
    config = _section(Config, values, "config")
    config.validate()
    return config
