"""Framework tests never load models, datasets, or use the network."""

from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from probe.artifacts import Artifacts
from probe.cli import main
from probe.config import load_config
from probe.pipeline import STAGES

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/base.yaml"


def test_overlays_and_overrides():
    config = load_config(
        [BASE, ROOT / "configs/probes/gemini.yaml", ROOT / "configs/losses/plugguard_atc.yaml"],
        ["probe.params.window_size=40", "evaluation.readouts=[last, max]"],
    )
    assert config.probe.name == "gemini"
    assert config.probe.params["window_size"] == 40
    assert config.loss.name == "plugguard_atc"
    assert config.evaluation.readouts == ["last", "max"]


def test_changing_probe_does_not_leak_old_parameters():
    config = load_config(
        [
            BASE,
            ROOT / "configs/probes/singprobe.yaml",
            ROOT / "configs/probes/gemini.yaml",
        ]
    )
    assert "sliding_window" not in config.probe.params
    assert config.probe.params["partial_prefix_windows"] is True


@pytest.mark.parametrize(
    "override",
    [
        "training.micro_batch_size=0",
        "training.epochs=1.5",
        "training.learning_rate=.nan",
        "features.block_indices=[19,19]",
        "data.split_ratios=[0.8,0.1,0.2]",
        "evaluation.readouts=[median]",
        "evaluation.readouts=[]",
        "name=../escape",
        "training.micro_bach_size=8",
        "seed=true",
        "data.max_samples=0",
        "data.max_samples=true",
        "data.max_samples=[5,10]",
        "data.max_samples={train: -1}",
        "data.max_samples={validation: 1.5}",
        "data.max_samples={validation: false}",
        "data.max_samples={typo: 5}",
        "features.splits=[]",
        "features.splits=[train, train]",
        "features.splits=[typo]",
        "features.splits=train",
    ],
)
def test_invalid_configuration_is_rejected(override):
    with pytest.raises(ValueError):
        load_config([BASE], [override])


def test_unknown_yaml_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("training:\n  micro_bach_size: 2\n")
    with pytest.raises(ValueError, match="Unknown keys"):
        load_config([BASE, path])


def test_artifact_reuse_and_invalidation():
    base = Artifacts.from_config(load_config([BASE]))
    different_loss = Artifacts.from_config(load_config([BASE], ["loss.name=singprobe_weighted"]))
    different_layer = Artifacts.from_config(load_config([BASE], ["features.block_indices=[19]"]))
    different_readout = Artifacts.from_config(load_config([BASE], ["evaluation.readouts=[last]"]))
    different_seed = Artifacts.from_config(load_config([BASE], ["seed=43"]))
    assert base.prepared == different_loss.prepared
    assert base.features == different_loss.features
    assert base.run != different_loss.run
    assert base.prepared == different_layer.prepared
    assert base.features != different_layer.features
    assert base.run == different_readout.run
    assert base.prepared == different_seed.prepared
    assert base.features == different_seed.features
    assert base.run != different_seed.run


def test_missing_component_name_is_a_configuration_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("probe:\n  params: {}\n")
    with pytest.raises(ValueError, match="Invalid probe"):
        load_config([BASE, path])


@pytest.mark.parametrize("stage", STAGES)
def test_dry_run_has_no_side_effects(stage, tmp_path, capsys):
    output = tmp_path / "untouched"
    assert main([stage, "--config", str(BASE), "--set", f"output_dir={output}", "--dry-run"]) == 0
    result = yaml.safe_load(capsys.readouterr().out)
    assert result["stage"] == stage
    assert result["status"] == "planned"
    assert not output.exists()


@pytest.mark.parametrize("stage", STAGES)
def test_stage_errors_are_reported(stage, tmp_path, caplog, monkeypatch):
    def fail(*args):
        raise ValueError("test-stage-error")

    monkeypatch.setattr("probe.cli.run", fail)
    output = tmp_path / "untouched"
    assert main([stage, "--config", str(BASE), "--set", f"output_dir={output}"]) == 1
    assert "test-stage-error" in caplog.text
    assert not output.exists()


def test_yaml_round_trip(tmp_path):
    config = load_config([BASE])
    path = tmp_path / "resolved.yaml"
    path.write_text(yaml.safe_dump(asdict(config)))
    assert load_config([path]) == config


def test_per_split_limit_overlays_and_cli_overrides():
    config = load_config([BASE, ROOT / "configs/pilot.yaml"], ["data.max_samples.train=100"])
    assert config.data.max_samples == {
        "train": 100,
        "validation": None,
        "beavertails_test": None,
        "wildguardtest": None,
    }
    assert config.data.split_seed == load_config([BASE]).data.split_seed
    assert config.data.split_ratios == load_config([BASE]).data.split_ratios


def test_old_pre_split_cap_artifacts_are_not_reused():
    from probe.artifacts import _key
    from probe.contracts import HEADS

    config = load_config([BASE], ["data.max_samples=512"])
    old_data_key = _key({"data": asdict(config.data), "heads": HEADS})
    assert Artifacts.from_config(config).prepared.name != old_data_key


def test_old_unanimous_label_caches_are_not_reused():
    from probe.artifacts import _key
    from probe.contracts import HEADS

    config = load_config([BASE])
    old_key = _key({"data": asdict(config.data), "heads": HEADS, "version": 2})
    assert Artifacts.from_config(config).prepared.name != old_key
