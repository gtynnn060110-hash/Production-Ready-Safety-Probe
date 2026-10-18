"""End-to-end stage tests with tiny local doubles; no network or 9B weights."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from probe.artifacts import Artifacts
from probe.config import ComponentConfig, load_config
from probe.data import CATEGORY_HEAD, EXCLUDED
from probe.features import BlockTap, encode_sample
from probe.io import checked_manifest, read_json
from probe.pipeline import run

ROOT = Path(__file__).resolve().parents[1]


class TinyTokenizer:
    is_fast = True
    pad_token_id = 0
    all_special_ids = []
    chat_template = "tiny-chat-v1"
    backend_tokenizer = SimpleNamespace(to_str=lambda: "tiny-tokenizer-v1")

    def apply_chat_template(self, messages, **kwargs):
        return (
            "<user>"
            + messages[0]["content"].strip()
            + "</user><assistant><think></think>"
            + messages[1]["content"]
            + "</assistant>"
        )

    def __call__(self, text, **kwargs):
        return {
            "input_ids": [ord(c) % 127 + 1 for c in text],
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }


class TinyTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(129, 8)
        self.layers = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
        self.norm = nn.LayerNorm(8)
        self.calls = 0

    def forward(self, input_ids, attention_mask, **kwargs):
        self.calls += 1
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


@pytest.fixture
def tiny_pipeline(tmp_path, monkeypatch):
    import datasets
    import transformers

    rows = []
    for i in range(100):
        active = [] if i % 2 == 0 else [list(CATEGORY_HEAD)[(i // 2) % len(CATEGORY_HEAD)]]
        rows.append(
            {
                "prompt": f"question {i}",
                "response": f"answer {i}",
                "is_safe": not bool(active),
                "category": {c: c in active for c in set(CATEGORY_HEAD) | EXCLUDED},
            }
        )
    benchmark = [
        {
            "prompt": f"external {i}",
            "response": f"response {i}",
            "response_harm_label": "harmful" if i % 2 else "unharmful",
        }
        for i in range(10)
    ]
    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda repo, *args, **kwargs: benchmark if "wildguard" in repo else rows,
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: TinyTokenizer()
    )
    text = TinyTextModel()
    model = SimpleNamespace(model=SimpleNamespace(language_model=text))
    model.eval = lambda: model
    model.requires_grad_ = lambda value: model
    monkeypatch.setattr(
        transformers.AutoModelForImageTextToText, "from_pretrained", lambda *a, **k: model
    )
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *a, **k: SimpleNamespace(
            text_config=SimpleNamespace(
                hidden_size=8, num_hidden_layers=2, max_position_embeddings=4096
            )
        ),
    )
    cfg = load_config(
        [ROOT / "configs/base.yaml"],
        [
            f"output_dir={tmp_path}",
            "features.block_indices=[0]",
            "features.device=cpu",
            "features.dtype=float32",
            "features.max_length=1000",
            "features.batch_size=2",
            "features.shard_tokens=1200",
            "training.device=cpu",
            "training.dtype=float32",
            "training.micro_batch_size=4",
            "training.gradient_accumulation_steps=3",
            "training.epochs=2",
            "evaluation.batch_size=4",
        ],
    )
    return cfg, text


def test_template_alignment_excludes_artificial_thinking_and_markers():
    tokenizer = TinyTokenizer()
    encoded = encode_sample(tokenizer, "question", " answer ")
    selected = "".join(
        c for c, keep in zip(encoded["rendered"], encoded["response_mask"], strict=True) if keep
    )
    assert selected == "answer"


def test_raw_hook_is_before_final_norm():
    text = TinyTextModel()
    tap = BlockTap(text, [0, 1])
    ids = torch.tensor([[1, 2, 3]])
    expected = text.layers[0](text.embed(ids))
    text(ids, torch.ones_like(ids))
    features = tap.collect()
    tap.close()
    torch.testing.assert_close(features[..., :8], expected)
    assert features.shape == (1, 3, 16)


def test_prepare_manifest_records_full_and_selected_counts(tiny_pipeline):
    config, _ = tiny_pipeline
    run("prepare", config)
    full_paths = Artifacts.from_config(config)
    full = read_json(full_paths.prepared / "manifest.json")
    config.data.max_samples = {"train": 13, "validation": 7}
    run("prepare", config)
    subset_paths = Artifacts.from_config(config)
    subset = read_json(subset_paths.prepared / "manifest.json")
    assert subset["full_splits"] == full["full_splits"]
    assert subset["splits"]["train"]["samples"] == 13
    assert subset["splits"]["validation"]["samples"] == 7
    assert subset["splits"]["beavertails_test"] == full["splits"]["beavertails_test"]
    assert subset["splits"]["wildguardtest"] == full["splits"]["wildguardtest"]
    from probe.io import read_jsonl

    for split in subset["splits"]:
        actual_ids = {r["id"] for r in read_jsonl(subset_paths.prepared / f"{split}.jsonl")}
        full_ids = {r["id"] for r in read_jsonl(full_paths.prepared / f"{split}.jsonl")}
        assert actual_ids <= full_ids


def test_train_validation_only_extraction_never_runs_test_forward(tiny_pipeline):
    config, text = tiny_pipeline
    config.features.splits = ["train", "validation"]
    paths = Artifacts.from_config(config)
    run("prepare", config)
    prepared = read_json(paths.prepared / "manifest.json")
    run("extract", config)
    manifest = read_json(paths.features / "manifest.json")
    assert set(manifest["splits"]) == {"train", "validation"}
    expected_calls = sum(
        (prepared["splits"][name]["samples"] + 1) // 2 for name in ["train", "validation"]
    )
    assert text.calls == expected_calls
    assert not (paths.features / "beavertails_test").exists()
    assert not (paths.features / "wildguardtest").exists()
    run("train", config)


def test_missing_requested_feature_split_is_rejected(tiny_pipeline):
    from probe.stages.extract import tokenize_splits

    config, _ = tiny_pipeline
    config.features.splits = ["train", "validation"]
    with pytest.raises(ValueError, match="not prepared"):
        tokenize_splits(
            config,
            Artifacts.from_config(config),
            {"splits": {"train": {"samples": 1}}},
            TinyTokenizer(),
        )


@pytest.mark.parametrize("probe", ["linear", "singprobe", "gemini"])
@pytest.mark.parametrize("loss", ["bce", "singprobe_weighted", "plugguard_atc"])
def test_four_stages_all_nine_combinations(tiny_pipeline, probe, loss):
    config, text = tiny_pipeline
    config.probe = ComponentConfig(probe)
    config.loss = ComponentConfig(loss)
    paths = Artifacts.from_config(config)
    run("prepare", config)
    run("extract", config)
    calls = text.calls
    run("extract", config)
    assert text.calls == calls  # verified cache reuse, no repeated backbone forward
    run("train", config)
    run("evaluate", config)
    checked_manifest(paths.run / "manifest.json")
    assert (paths.run / "checkpoints/best.pt").exists()
    for mode in ["last", "mean", "max"]:
        result = read_json(paths.run / "evaluation" / f"{mode}.json")
        assert result["threshold_source"].startswith("BeaverTails validation only")
        assert result["splits"]["wildguardtest"]["privacy"]["n"] == 0
        assert result["splits"]["wildguardtest"]["overall"]["n"] == 10
        assert result["splits"]["beavertails_test"]["overall"]["n"] > 0
    assert read_json(paths.run / "config.yaml") == asdict(config)


def test_extraction_resume_and_corruption_detection(tiny_pipeline):
    config, text = tiny_pipeline
    paths = Artifacts.from_config(config)
    run("prepare", config)
    run("extract", config)
    manifest = read_json(paths.features / "manifest.json")
    (paths.features / "manifest.json").unlink()  # simulated crash after final shard commit
    calls = text.calls
    run("extract", config)
    assert text.calls == calls
    tensor = paths.features / manifest["splits"]["train"]["shards"][0]["tensor"]
    with tensor.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        run("train", config)


def test_epoch_resume_uses_existing_checkpoint(tiny_pipeline):
    config, _ = tiny_pipeline
    paths = Artifacts.from_config(config)
    for stage in ["prepare", "extract", "train"]:
        run(stage, config)
    previous = (paths.run / "checkpoints/last.pt").read_bytes()
    (paths.run / "manifest.json").unlink()  # simulate crash before completion marker
    run("train", config)
    assert (paths.run / "checkpoints/last.pt").read_bytes() == previous


def test_interrupted_epoch_resume_matches_uninterrupted_training(tiny_pipeline, monkeypatch):
    from probe.stages import train

    config, _ = tiny_pipeline
    paths = Artifacts.from_config(config)
    for stage in ["prepare", "extract", "train"]:
        run(stage, config)
    reference = torch.load(paths.run / "checkpoints/last.pt", weights_only=True)
    config.name = "interrupted"
    resumed_paths = Artifacts.from_config(config)
    original_save = train.save_checkpoint

    def save_then_interrupt(path, state):
        original_save(path, state)
        if path.name == "last.pt" and state["epoch"] == 0:
            raise RuntimeError("simulated crash after first epoch")

    monkeypatch.setattr(train, "save_checkpoint", save_then_interrupt)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run("train", config)
    monkeypatch.setattr(train, "save_checkpoint", original_save)
    run("train", config)
    resumed = torch.load(resumed_paths.run / "checkpoints/last.pt", weights_only=True)
    assert resumed["history"] == reference["history"]
    for name in resumed["model"]:
        torch.testing.assert_close(resumed["model"][name], reference["model"][name], rtol=0, atol=0)


def test_budget_check_stops_before_weight_loading(tiny_pipeline, monkeypatch):
    import transformers

    config, _ = tiny_pipeline
    config.features.max_cache_gb = 1e-12
    paths = Artifacts.from_config(config)

    def forbidden(*args, **kwargs):
        pytest.fail("Weight loading must not happen over budget")

    monkeypatch.setattr(transformers.AutoModelForImageTextToText, "from_pretrained", forbidden)
    run("prepare", config)
    with pytest.raises(ValueError, match="exceeds features.max_cache_gb"):
        run("extract", config)
    assert not list(paths.features.rglob("*.safetensors"))


def test_extraction_keeps_validation_and_test_overlength_samples(tmp_path):
    from probe.config import Config
    from probe.io import read_jsonl, write_jsonl
    from probe.stages.extract import tokenize_splits

    config = Config(output_dir=str(tmp_path))
    config.features.max_length = 100
    paths = Artifacts.from_config(config)
    short = {
        "id": "short",
        "prompt": "q",
        "response": "a",
        "labels": [0.0] * 5,
        "label_mask": [True] * 5,
    }
    long = short | {"id": "long", "response": "x" * 200}
    for split in ["train", "validation", "beavertails_test"]:
        write_jsonl(paths.prepared / f"{split}.jsonl", [short, long])
    stats, _ = tokenize_splits(
        config,
        paths,
        {"splits": {s: {"samples": 2} for s in ["train", "validation", "beavertails_test"]}},
        TinyTokenizer(),
    )
    assert stats["train"]["samples"] == 1
    assert stats["train"]["dropped_overlength"] == 1
    assert stats["validation"]["samples"] == stats["beavertails_test"]["samples"] == 2
    assert len(list(read_jsonl(paths.features / "tokens/validation.jsonl"))) == 2
