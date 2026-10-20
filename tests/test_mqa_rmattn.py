from dataclasses import replace
from pathlib import Path

import pytest
import torch

from probe.config import ComponentConfig, load_config
from probe.contracts import FeatureBatch
from probe.probes import build_probe
from probe.sweep import trial_configs


def features(x, response=None, attention=None):
    b, t, _ = x.shape
    mask = torch.ones(b, t, dtype=torch.bool, device=x.device)
    return FeatureBatch(
        [str(i) for i in range(b)], x,
        mask if attention is None else attention,
        mask if response is None else response,
        torch.ones(b, 5, device=x.device), mask.new_ones((b, 5)),
    )


def hybrid(context=3, risk=3):
    return build_probe(ComponentConfig("mqa_rmattn", {
        "context_window": context, "risk_window": risk, "num_query_heads": 2,
        "head_dim": 4, "mlp_hidden_sizes": [12, 12], "internal_heads": 2,
        "attention_chunk_size": 2,
    }), input_dim=8, num_heads=5)


def test_stacked_receptive_field_and_prompt_context():
    torch.manual_seed(7)
    model = hybrid(context=3, risk=2).eval()
    x = torch.randn(1, 12, 8)
    base = model(features(x)).local_logits[:, -1]
    outside = x.clone()
    outside[:, :8] += 20  # Last score can only depend on positions 8..11.
    torch.testing.assert_close(base, model(features(outside)).local_logits[:, -1])
    inside = x.clone()
    inside[:, 8] += 20
    assert not torch.allclose(base, model(features(inside)).local_logits[:, -1])
    response = torch.tensor([[False] * 3 + [True] * 9])
    altered_prompt = x.clone()
    altered_prompt[:, :3] += 20
    assert not torch.allclose(
        model(features(x, response)).local_logits[:, 3],
        model(features(altered_prompt, response)).local_logits[:, 3],
    )


def test_risk_pooling_matches_window_math_on_context_features():
    torch.manual_seed(13)
    model = hybrid().eval()
    response = torch.tensor([[False, False, True, True, False, True, True, True]])
    data = features(torch.randn(1, 8, 8), response)
    context = model.context(data)
    # Risk pooling receives the full latent width; no intermediate class bottleneck.
    assert context.shape == (1, 8, 8)
    assert not hasattr(model.context, "classifier")
    y = model.risk.mlp(context)
    scores = model.risk.query(y).reshape(1, 8, 2, 5)
    values = model.risk.value(y).reshape(1, 8, 2, 5)
    result = model(data)
    assert torch.equal(result.local_mask, response)
    for t in (2, 3, 5, 6, 7):
        left = t
        while left > max(0, t - 2) and response[0, left - 1]:
            left -= 1
        expected = (
            scores[:, left:t + 1].softmax(1) * values[:, left:t + 1]
        ).sum(1).sum(1) + model.risk.bias
        torch.testing.assert_close(result.local_logits[:, t], expected)


def test_full_context_equals_large_window_with_identical_weights():
    torch.manual_seed(19)
    full, window = hybrid(context=0), hybrid(context=2048)
    window.load_state_dict(full.state_dict())
    data = features(torch.randn(2, 13, 8))
    torch.testing.assert_close(full(data).local_logits, window(data).local_logits)


def test_standalone_singprobe_keeps_legacy_checkpoint_keys():
    model = build_probe(ComponentConfig("singprobe"), input_dim=8, num_heads=5)
    assert set(model.state_dict()) == {
        "proj_q.weight", "proj_k.weight", "proj_v.weight", "o_proj.weight",
        "norm.weight", "classifier.weight", "classifier.bias",
    }
    classifier = model.classifier
    captured = []
    handle = model.norm.register_forward_hook(lambda module, args, out: captured.append(out))
    data = features(torch.randn(1, 8, 8))
    result = model(data)
    handle.remove()
    torch.testing.assert_close(result.local_logits, classifier(captured[0]))


@pytest.mark.parametrize("params", [
    {"context_window": -1}, {"context_window": True}, {"risk_window": 0},
    {"risk_window": 1.5}, {"num_kv_heads": 2}, {"partial_prefix_windows": False},
])
def test_invalid_stacked_settings_are_rejected(params):
    with pytest.raises(ValueError):
        build_probe(ComponentConfig("mqa_rmattn", params), input_dim=8, num_heads=5)


def test_search_axes_are_independent_and_keep_source_cache():
    from probe.artifacts import Artifacts

    root = Path(__file__).resolve().parents[1]
    config = load_config([
        root / "configs/base.yaml", root / "configs/probes/mqa_rmattn.yaml",
        root / "configs/sweeps/mqa_rmattn_weighted.yaml",
    ])
    candidates = list(trial_configs(config, [64, 256, 2048], [10, 20, 40], [5e-5, 1e-4, 3e-4]))
    assert len(candidates) == 27
    assert len({c.name for c in candidates}) == 27
    assert len({Artifacts.from_config(c).features for c in candidates}) == 1
    assert config.probe.params["context_window"] == 256
    assert config.probe.params["risk_window"] == 20


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_bfloat16_backward_and_padding():
    from probe.losses import build_loss

    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.manual_seed(29)
    model = hybrid().cuda()
    data = features(torch.randn(2, 11, 8, device="cuda"))
    attention = data.attention_mask.clone()
    attention[1, 7:] = False
    data = replace(data, attention_mask=attention, response_mask=attention)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = model(data)
    loss = build_loss(ComponentConfig("singprobe_weighted"))(
        result, data.labels, data.label_mask
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert not result.local_mask[1, 7:].any()
