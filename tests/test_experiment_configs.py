"""Formal configurations stay full-data, cache-compatible, and pair-specifically tuned."""

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from probe.artifacts import Artifacts
from probe.config import load_config
from probe.contracts import FeatureBatch
from probe.losses import build_loss
from probe.pipeline import STAGES, plan
from probe.probes import build_probe

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/base.yaml"
EXPERIMENTS = ROOT / "configs/experiments"
# filename -> (probe, loss, learning rate, window, temperature, anchors, regularization)
EXPECTED = {
    "linear_bce": ("linear", "bce", 3e-4, None, None, None, None),
    "linear_weighted": ("linear", "singprobe_weighted", 3e-4, None, 0.1, None, None),
    "linear_atc": ("linear", "plugguard_atc", 3e-4, None, None, 5, 0.0),
    "singprobe_bce": ("singprobe", "bce", 1e-4, 64, None, None, None),
    "singprobe_weighted": ("singprobe", "singprobe_weighted", 3e-4, 64, 0.1, None, None),
    "singprobe_atc": ("singprobe", "plugguard_atc", 3e-4, 256, None, 5, 0.1),
    "gemini_bce": ("gemini", "bce", 3e-4, 40, None, None, None),
    "gemini_weighted": ("gemini", "singprobe_weighted", 3e-4, 40, 0.1, None, None),
    "gemini_atc": ("gemini", "plugguard_atc", 3e-4, 40, None, 1, 0.0),
}


def configuration(name, overrides=None):
    return load_config([BASE, EXPERIMENTS / f"{name}.yaml"], overrides)


def test_exactly_nine_formal_combinations():
    assert {p.stem for p in EXPERIMENTS.glob("*.yaml")} == set(EXPECTED)


@pytest.mark.parametrize("name", EXPECTED)
def test_formal_config_has_full_data_and_shared_protocol(name):
    config = configuration(name)
    assert config.name == f"full_{name}"
    assert config.output_dir == "artifacts"
    assert config.data.max_samples is None
    assert config.data.benchmark_enabled
    assert config.data.source_splits == ["330k_train", "330k_test"]
    assert config.data.split_seed == 42
    assert config.data.split_ratios == [0.8, 0.1, 0.1]
    assert config.features.block_indices == [15]
    assert config.features.splits is None
    assert config.features.normalize == "rms_no_affine"
    assert config.training.epochs == 4
    assert config.training.micro_batch_size == 8
    assert config.training.gradient_accumulation_steps == 4
    assert config.training.selection_metric == "overall_auprc"
    assert config.training.selection_readout == "max"
    assert config.evaluation.targets == ["beavertails_test", "wildguardtest"]
    assert config.evaluation.readouts == ["last", "mean", "max"]
    assert (
        config.data.train_revision and config.data.benchmark_revision and config.features.revision
    )
    for stage in STAGES:
        assert plan(stage, config)["status"] == "planned"


@pytest.mark.parametrize("name", EXPECTED)
def test_pair_specific_parameters_match_selected_candidates(name):
    config = configuration(name)
    probe, loss, lr, window, temperature, anchors, regularization = EXPECTED[name]
    assert config.probe.name == probe
    assert config.loss.name == loss
    assert config.training.learning_rate == lr
    if probe == "linear":
        assert config.probe.params == {}
    elif probe == "singprobe":
        assert config.probe.params == {
            "num_query_heads": 4,
            "num_kv_heads": 1,
            "head_dim": 64,
            "sliding_window": window,
            "rms_norm_eps": 1e-6,
            "init_bias": 0.0,
        }
    else:
        assert config.probe.params == {
            "mlp_hidden_sizes": [100, 100],
            "internal_heads": 10,
            "window_size": window,
            "partial_prefix_windows": True,
        }
    if loss == "bce":
        assert config.loss.params == {}
    elif loss == "singprobe_weighted":
        assert config.loss.params == {
            "temperature": temperature,
            "beta_tau": 0.35,
            "detach_weights": True,
        }
    else:
        assert config.loss.params == {
            "anchor_tokens": anchors,
            "lambda_tv": regularization,
            "lambda_mono": regularization,
        }


@pytest.mark.parametrize("name", EXPECTED)
def test_selected_components_build_and_backpropagate(name):
    config = configuration(name)
    model = build_probe(config.probe, input_dim=8, num_heads=5)
    objective = build_loss(config.loss)
    data = FeatureBatch(
        ids=["a", "b"],
        hidden_states=torch.randn(2, 12, 8),
        attention_mask=torch.ones(2, 12, dtype=torch.bool),
        response_mask=torch.tensor([[False] * 2 + [True] * 10] * 2),
        labels=torch.tensor([[0.0] * 5, [1.0] * 5]),
        label_mask=torch.ones(2, 5, dtype=torch.bool),
    )
    value = objective(model(data), data.labels, data.label_mask)
    value.backward()
    assert torch.isfinite(value)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_all_combinations_and_seeds_reuse_data_and_features_but_not_runs():
    configurations = [
        configuration(name, [f"seed={seed}"]) for name in EXPECTED for seed in [42, 43, 44]
    ]
    paths = [Artifacts.from_config(c) for c in configurations]
    assert len({p.prepared for p in paths}) == 1
    assert len({p.features for p in paths}) == 1
    assert len({p.run for p in paths}) == 27


def test_component_overlays_do_not_carry_parameters_from_previous_pair():
    config = load_config(
        [BASE, EXPERIMENTS / "singprobe_atc.yaml", EXPERIMENTS / "linear_bce.yaml"]
    )
    assert asdict(config) == asdict(configuration("linear_bce"))
