"""Causal context MQA and response-only rolling risk attention.

The default outputs are R01 through R19, then overall risk. Input features are
raw cached hidden states; this module applies FP32 RMS normalization once.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _integer(name, value, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


class _CausalMQA(nn.Module):
    def __init__(self, input_dim, num_query_heads, head_dim, window, chunk, eps):
        super().__init__()
        self.heads, self.dim = num_query_heads, head_dim
        self.window, self.chunk = window, chunk
        width = num_query_heads * head_dim
        self.proj_q = nn.Linear(input_dim, width, bias=False)
        self.proj_k = nn.Linear(input_dim, head_dim, bias=False)
        self.proj_v = nn.Linear(input_dim, head_dim, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)
        self.norm = nn.RMSNorm(width, eps=eps)

    def _block(self, q, k, v, attention_mask, start, end):
        left = max(0, start - self.window + 1) if self.window else 0
        qi = torch.arange(start, end, device=q.device)
        ki = torch.arange(left, end, device=q.device)
        allowed = ki[None, :] <= qi[:, None]
        if self.window:
            allowed = allowed & (ki[None, :] > qi[:, None] - self.window)
        allowed = allowed[None, None] & attention_mask[:, None, None, left:end]
        return F.scaled_dot_product_attention(
            q[:, :, start:end], k[:, :, left:end], v[:, :, left:end],
            attn_mask=allowed, enable_gqa=self.heads != 1,
        )

    def forward(self, x, attention_mask):
        batch, length, _ = x.shape
        query = self.proj_q(x)
        q = query.view(batch, length, self.heads, self.dim).transpose(1, 2)
        k, v = self.proj_k(x).unsqueeze(1), self.proj_v(x).unsqueeze(1)
        blocks = []
        for start in range(0, length, self.chunk):
            args = (q, k, v, attention_mask, start, min(start + self.chunk, length))
            if self.training and torch.is_grad_enabled():
                blocks.append(checkpoint(self._block, *args, use_reentrant=False))
            else:
                blocks.append(self._block(*args))
        mixed = torch.cat(blocks, dim=2).transpose(1, 2).reshape(batch, length, -1)
        return self.norm(query + self.o_proj(mixed))


class _RollingRiskAttention(nn.Module):
    def __init__(self, input_dim, classes, hidden_sizes, heads, window, chunk):
        super().__init__()
        self.heads, self.classes = heads, classes
        self.window, self.chunk = window, chunk
        first, second = hidden_sizes
        self.mlp = nn.Sequential(nn.Linear(input_dim, first), nn.ReLU(), nn.Linear(first, second))
        self.query = nn.Linear(second, heads * classes, bias=False)
        self.value = nn.Linear(second, heads * classes, bias=False)
        self.bias = nn.Parameter(torch.zeros(classes))

    def _block(self, scores, values, mask, spans, start, end):
        # Recompute only this block's rolling attention in backward. Keeping
        # all T * window * heads * classes weights would dominate probe memory.
        left = max(0, start - self.window + 1)
        padding = max(0, self.window - 1 - start)
        scores = F.pad(scores[:, left:end], (0, 0, 0, 0, padding, 0)).unfold(1, self.window, 1)
        values = F.pad(values[:, left:end], (0, 0, 0, 0, padding, 0)).unfold(1, self.window, 1)
        window_spans = F.pad(spans[:, left:end], (padding, 0), value=-1).unfold(1, self.window, 1)
        valid = F.pad(mask[:, left:end], (padding, 0), value=False).unfold(1, self.window, 1)
        valid = valid & (window_spans == spans[:, start:end, None]) & mask[:, start:end, None]
        valid = valid[:, :, None, None, :]
        # Finite fill keeps all-masked positions and their gradients finite.
        weights = scores.masked_fill(~valid, -1e30).softmax(-1) * valid
        return (weights * values).sum(-1).sum(2)

    def forward(self, x, mask):
        y = self.mlp(x)
        batch, length, _ = y.shape
        scores = self.query(y).view(batch, length, self.heads, self.classes)
        values = self.value(y).view(batch, length, self.heads, self.classes)
        starts = mask & ~F.pad(mask[:, :-1], (1, 0), value=False)
        spans = starts.long().cumsum(1)
        blocks = []
        for start in range(0, length, self.chunk):
            args = (scores, values, mask, spans, start, min(start + self.chunk, length))
            if self.training and torch.is_grad_enabled():
                blocks.append(checkpoint(self._block, *args, use_reentrant=False))
            else:
                blocks.append(self._block(*args))
        return torch.cat(blocks, dim=1) + self.bias


class Probe(nn.Module):
    """Full causal context -> optional residual FFN -> local risk logits.

    ``context_window=0`` attends to all preceding real tokens, including prompt
    and image tokens. Risk windows contain only the current contiguous response
    span. The two masks must be boolean tensors on the feature tensor's device.
    Keep parameters in FP32; forward disables autocast for stable probe training.
    """

    def __init__(self, input_dim=4096, ffn=0, num_heads=20, context_window=0,
                 risk_window=40, num_query_heads=4, head_dim=64, internal_heads=10,
                 attention_chunk_size=256, mlp_hidden_sizes=(100, 100), rms_norm_eps=1e-6):
        super().__init__()
        for name, value in (('input_dim', input_dim), ('num_heads', num_heads),
                            ('risk_window', risk_window), ('num_query_heads', num_query_heads),
                            ('head_dim', head_dim), ('internal_heads', internal_heads),
                            ('attention_chunk_size', attention_chunk_size)):
            _integer(name, value)
        _integer('ffn', ffn, 0)
        _integer('context_window', context_window, 0)
        if not isinstance(mlp_hidden_sizes, (list, tuple)) or len(mlp_hidden_sizes) != 2:
            raise ValueError('mlp_hidden_sizes must contain two positive integers')
        for value in mlp_hidden_sizes:
            _integer('mlp_hidden_sizes entry', value)
        if (not isinstance(rms_norm_eps, (int, float)) or isinstance(rms_norm_eps, bool)
                or not math.isfinite(rms_norm_eps) or rms_norm_eps <= 0):
            raise ValueError('rms_norm_eps must be positive and finite')
        self._config = dict(input_dim=input_dim, ffn=ffn, num_heads=num_heads,
                            context_window=context_window, risk_window=risk_window,
                            num_query_heads=num_query_heads, head_dim=head_dim,
                            internal_heads=internal_heads, attention_chunk_size=attention_chunk_size,
                            mlp_hidden_sizes=tuple(mlp_hidden_sizes), rms_norm_eps=rms_norm_eps)
        self.context = _CausalMQA(input_dim, num_query_heads, head_dim, context_window,
                                  attention_chunk_size, rms_norm_eps)
        width = num_query_heads * head_dim
        if ffn:
            self.context_ffn = nn.Sequential(nn.Linear(width, ffn), nn.GELU(), nn.Linear(ffn, width))
            self.context_ffn_norm = nn.RMSNorm(width, eps=rms_norm_eps)
        else:
            self.context_ffn = None
        self.risk = _RollingRiskAttention(width, num_heads, mlp_hidden_sizes, internal_heads,
                                          risk_window, attention_chunk_size)

    def get_config(self):
        """Constructor arguments sufficient to rebuild this checkpoint."""
        return dict(self._config)

    def forward(self, hidden_states, attention_mask, response_mask):
        if (not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3
                or hidden_states.shape[-1] != self._config['input_dim']
                or not hidden_states.is_floating_point()):
            raise ValueError('hidden_states must be floating point [B, T, input_dim]')
        for name, mask in (('attention_mask', attention_mask), ('response_mask', response_mask)):
            if (not isinstance(mask, torch.Tensor) or mask.shape != hidden_states.shape[:2]
                    or mask.dtype != torch.bool or mask.device != hidden_states.device):
                raise ValueError(f'{name} must be boolean [B, T] on the hidden_states device')
        if self.context.proj_q.weight.dtype != torch.float32:
            raise ValueError('Probe parameters must stay in FP32; pass raw FP16/BF16 features directly')
        local_mask = response_mask & attention_mask
        if not hidden_states.shape[0] or not hidden_states.shape[1]:
            return hidden_states.new_zeros((*hidden_states.shape[:2], self._config['num_heads']),
                                           dtype=torch.float32), local_mask
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            # Clear padding before normalization: even NaN padding cannot affect
            # real keys/values or the response windows' backward pass.
            x = hidden_states.float().masked_fill(~attention_mask.unsqueeze(-1), 0)
            x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self._config['rms_norm_eps'])
            x = self.context(x, attention_mask)
            if self.context_ffn is not None:
                x = self.context_ffn_norm(x + self.context_ffn(x))
            return self.risk(x, local_mask), local_mask
