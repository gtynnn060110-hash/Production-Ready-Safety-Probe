"""Architecture adapter checks with a randomly initialized, tiny hybrid Qwen."""

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from probe.features import BlockTap


def test_hybrid_qwen_raw_tap_causality_and_right_padding():
    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [2, 3, 3],
        },
    )
    config._attn_implementation = "sdpa"
    model = Qwen3_5TextModel(config).eval()
    tap = BlockTap(model, [0, 1])
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    with torch.no_grad():
        result = model(ids, attention_mask=torch.ones_like(ids), use_cache=False)
        full = tap.collect()
        model(ids[:, :4], attention_mask=torch.ones_like(ids[:, :4]), use_cache=False)
        prefix = tap.collect()
        model(
            torch.cat([ids, ids], 0),
            attention_mask=torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]]),
            use_cache=False,
        )
        padded = tap.collect()
    tap.close()
    torch.testing.assert_close(full[:, :4], prefix, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(padded[1:2, :4], prefix, rtol=1e-4, atol=1e-5)
    assert not torch.allclose(full[:, :, 32:], result.last_hidden_state)
