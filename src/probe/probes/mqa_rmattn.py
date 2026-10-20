"""Causal MQA context encoding followed by response-only rolling risk attention.

The first stage exposes features, not class logits. The second stage uses fixed
learned queries per class/head. As in GeminiProbe, internal heads are summed at
each position before a separate last/mean/max temporal readout.
"""

from dataclasses import replace

from torch import nn

from ..contracts import FeatureBatch, ProbeOutput
from .gemini import GeminiProbe
from .singprobe import CausalMQA


class MQARMAttnProbe(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        context_window: int = 256,
        risk_window: int = 20,
        num_query_heads: int = 4,
        num_kv_heads: int = 1,
        head_dim: int = 64,
        rms_norm_eps: float = 1e-6,
        attention_chunk_size: int = 256,
        mlp_hidden_sizes=(100, 100),
        internal_heads: int = 10,
        partial_prefix_windows: bool = True,
    ):
        super().__init__()
        if type(context_window) is not int or context_window < 0:
            raise ValueError("context_window must be a non-negative integer; 0 means full causal")
        if type(risk_window) is not int or risk_window < 1:
            raise ValueError("risk_window must be a positive integer")
        self.context = CausalMQA(
            input_dim=input_dim,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            sliding_window=context_window,
            rms_norm_eps=rms_norm_eps,
            attention_chunk_size=attention_chunk_size,
        )
        self.risk = GeminiProbe(
            input_dim=num_query_heads * head_dim,
            num_heads=num_heads,
            mlp_hidden_sizes=mlp_hidden_sizes,
            internal_heads=internal_heads,
            window_size=risk_window,
            partial_prefix_windows=partial_prefix_windows,
        )

    def forward(self, batch: FeatureBatch) -> ProbeOutput:
        contextual = replace(batch, hidden_states=self.context(batch))
        return self.risk(contextual)
