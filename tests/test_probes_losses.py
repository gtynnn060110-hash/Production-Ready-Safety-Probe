import pytest
import torch
from torch.nn import functional as F

from probe.config import ComponentConfig
from probe.contracts import FeatureBatch, ProbeOutput
from probe.losses import build_loss
from probe.losses.token import PlugGuardATC, SingProbeWeighted, TokenBCE
from probe.metrics import readout
from probe.probes import build_probe


def batch(x, response=None, attention=None):
    b, t, _ = x.shape
    mask = torch.ones(b, t, dtype=torch.bool)
    return FeatureBatch(
        [str(i) for i in range(b)],
        x,
        mask if attention is None else attention,
        mask if response is None else response,
        torch.ones(b, 5),
        torch.ones(b, 5, dtype=torch.bool),
    )


@pytest.mark.parametrize("probe", ["linear", "singprobe", "gemini", "mqa_rmattn"])
def test_causality_and_padding(probe):
    torch.manual_seed(42)
    params = {"sliding_window": 3, "attention_chunk_size": 2} if probe == "singprobe" else {}
    if probe == "mqa_rmattn":
        params = {"context_window": 3, "risk_window": 3, "attention_chunk_size": 2}
    model = build_probe(ComponentConfig(probe, params), input_dim=8, num_heads=5).eval()
    x = torch.randn(1, 9, 8)
    mask = torch.tensor([[False, False] + [True] * 7])
    full = model(batch(x, mask)).local_logits
    prefix = model(batch(x[:, :6], mask[:, :6])).local_logits
    torch.testing.assert_close(full[:, :6], prefix, rtol=1e-5, atol=1e-6)
    padded = torch.cat([x[:, :6], torch.randn(1, 3, 8) * 100], 1)
    attention = torch.tensor([[True] * 6 + [False] * 3])
    padded_result = model(batch(padded, mask & attention, attention)).local_logits
    torch.testing.assert_close(prefix, padded_result[:, :6], rtol=1e-5, atol=1e-6)


def test_singprobe_window_and_query_residual():
    model = build_probe(
        ComponentConfig("singprobe", {"sliding_window": 2}), input_dim=8, num_heads=5
    )
    x = torch.randn(1, 6, 8)
    altered = x.clone()
    altered[:, :3] += 100
    torch.testing.assert_close(
        model(batch(x)).local_logits[:, -1], model(batch(altered)).local_logits[:, -1]
    )
    with torch.no_grad():
        model.o_proj.weight.zero_()
    torch.testing.assert_close(
        model(batch(x)).local_logits, model.classifier(model.norm(model.proj_q(x)))
    )


def test_gemini_window_matches_explicit_loop_and_ignores_prompt():
    model = build_probe(ComponentConfig("gemini", {"window_size": 3}), input_dim=8, num_heads=5)
    x = torch.randn(1, 8, 8)
    response = torch.tensor([[False, False, True, True, False, True, True, True]])
    out = model(batch(x, response))
    y = model.mlp(x)
    a = model.query(y).reshape(1, 8, model.heads, 5)
    v = model.value(y).reshape(1, 8, model.heads, 5)
    for t in [2, 3, 5, 6, 7]:
        left = t
        while left > max(0, t - 2) and response[0, left - 1]:
            left -= 1
        expected = (a[:, left : t + 1].softmax(1) * v[:, left : t + 1]).sum(1).sum(1) + model.bias
        torch.testing.assert_close(out.local_logits[:, t], expected)


def test_gemini_window_one_is_pointwise_value():
    model = build_probe(
        ComponentConfig("gemini", {"window_size": 1, "internal_heads": 1}), input_dim=8, num_heads=5
    )
    x = torch.randn(2, 5, 8)
    torch.testing.assert_close(model(batch(x)).local_logits, model.value(model.mlp(x)) + model.bias)


@pytest.mark.parametrize("probe", ["linear", "singprobe", "gemini", "mqa_rmattn"])
@pytest.mark.parametrize("loss", ["bce", "singprobe_weighted", "plugguard_atc"])
def test_all_nine_combinations_have_finite_gradients(probe, loss):
    model = build_probe(ComponentConfig(probe), input_dim=8, num_heads=5)
    data = batch(torch.randn(2, 7, 8))
    data.labels[0] = 0
    value = build_loss(ComponentConfig(loss))(model(data), data.labels, data.label_mask)
    value.backward()
    assert torch.isfinite(value)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize("loss", [TokenBCE(), SingProbeWeighted(), PlugGuardATC()])
def test_empty_masks_padding_and_unknown_heads(loss):
    z = torch.randn(2, 7, 5, requires_grad=True)
    mask = torch.tensor([[False, True, True, True, False, False, False], [False] * 7])
    labels = torch.zeros(2, 5)
    known = torch.tensor([[True, False, True, True, True], [False] * 5])
    a = loss(ProbeOutput(z, mask), labels, known)
    b = loss(ProbeOutput(z[:1, :4], mask[:1, :4]), labels[:1], known[:1])
    torch.testing.assert_close(a, b)
    empty = loss(ProbeOutput(z, torch.zeros_like(mask)), labels, torch.zeros_like(known))
    assert float(empty.detach()) == 0.0
    (a + empty).backward()
    assert torch.isfinite(z.grad).all()
    assert (z.grad[:, :, 1] == 0).all()


def test_uniform_bce_is_not_bce_of_mean_logits():
    z = torch.tensor([[[-4.0], [4.0]]], requires_grad=True)
    mask = torch.ones(1, 2, dtype=torch.bool)
    y, known = torch.ones(1, 1), torch.ones(1, 1, dtype=torch.bool)
    value = TokenBCE()(ProbeOutput(z, mask), y, known)
    torch.testing.assert_close(value, F.binary_cross_entropy_with_logits(z, torch.ones_like(z)))
    assert value > F.binary_cross_entropy_with_logits(z.mean(1), y)


def test_weighted_beta_population_std_and_detach():
    z = torch.tensor([[[-3.0], [0.0], [2.0]], [[-2.0], [1.0], [4.0]]], requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    known = torch.ones(2, 1, dtype=torch.bool)
    output = ProbeOutput(z, mask)
    detached = SingProbeWeighted()
    weights = detached.confidence_weights(output, known)
    torch.testing.assert_close(weights.sum(1), torch.ones(2, 1))
    assert not weights.requires_grad
    assert (weights[~mask] == 0).all()
    expected = (z.sigmoid()[mask].std(unbiased=False) / 0.35).clamp(0, 1)
    assert detached.diagnostics["beta"][0] == pytest.approx(expected.item())
    attached = SingProbeWeighted(detach_weights=False)
    assert attached.confidence_weights(output, known).requires_grad
    y = torch.ones(2, 1)
    gd = torch.autograd.grad(detached(output, y, known), z, retain_graph=True)[0]
    ga = torch.autograd.grad(attached(output, y, known), z)[0]
    assert not torch.allclose(gd, ga)


def test_zero_beta_is_uniform_and_has_finite_gradient():
    z = torch.zeros(1, 3, 1, requires_grad=True)
    out = ProbeOutput(z, torch.ones(1, 3, dtype=torch.bool))
    known = torch.ones(1, 1, dtype=torch.bool)
    objective = SingProbeWeighted(detach_weights=False)
    torch.testing.assert_close(objective.confidence_weights(out, known), torch.full_like(z, 1 / 3))
    objective(out, torch.ones(1, 1), known).backward()
    assert torch.isfinite(z.grad).all()


@pytest.mark.parametrize("length", [1, 2, 3, 7, 20, 25])
def test_atc_short_anchors_against_reference(length):
    z = torch.linspace(-2, 2, length).reshape(1, length, 1).requires_grad_()
    out = ProbeOutput(z, torch.ones(1, length, dtype=torch.bool))
    y, known = torch.ones(1, 1), torch.ones(1, 1, dtype=torch.bool)
    n = min(10, length // 2)
    if length == 1:
        expected = F.softplus(-z).mean()
    else:
        expected = 0.5 * (F.softplus(z[:, :n]).mean() + F.softplus(-z[:, -n:]).mean())
        delta = z.sigmoid()[:, 1:] - z.sigmoid()[:, :-1]
        expected = expected + 0.01 * delta.abs().mean() + 0.01 * (-delta).relu().mean()
    torch.testing.assert_close(PlugGuardATC()(out, y, known), expected)


def test_atc_does_not_connect_disjoint_spans():
    z = torch.tensor([[[4.0], [99.0], [-4.0]]])
    out = ProbeOutput(z, torch.tensor([[True, False, True]]))
    y, known = torch.ones(1, 1), torch.ones(1, 1, dtype=torch.bool)
    torch.testing.assert_close(
        PlugGuardATC(lambda_tv=1, lambda_mono=1)(out, y, known),
        PlugGuardATC(lambda_tv=0, lambda_mono=0)(out, y, known),
    )


def test_readouts_ignore_prompt_padding_and_mean_logits():
    out = ProbeOutput(
        torch.tensor([[[100.0], [-3.0], [1.0], [99.0]]]), torch.tensor([[False, True, True, False]])
    )
    assert readout(out, "last").item() == 1
    assert readout(out, "max").item() == 1
    assert readout(out, "mean").item() == -1
