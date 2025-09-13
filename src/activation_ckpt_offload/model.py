from __future__ import annotations

import math

import torch
from torch import nn

# flash attention only works on Ampere GPU. Not working on free-for-use T4
_HAS_FLASH = False
try:
    from flash_attn import flash_attn_varlen_qkvpacked_func as _fa_varlen_qkvpacked

    _HAS_FLASH = True
except Exception:
    _HAS_FLASH = False


def _varlen_attn_reference(qkv: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Differentiable O(L^2) varlen attention fallback for CPU / GPUs without FlashAttn."""
    t, three, n_h, h_d = qkv.shape
    assert three == 3
    out = qkv.new_empty((t, n_h, h_d))
    bb = cu_seqlens.numel() - 1
    scale = 1.0 / math.sqrt(h_d)

    for b in range(bb):
        s = int(cu_seqlens[b])
        e = int(cu_seqlens[b + 1])
        if e <= s:
            continue
        q = qkv[s:e, 0]
        k = qkv[s:e, 1]
        v = qkv[s:e, 2]
        scores = torch.einsum("lhd,shd->hls", q, k) * scale
        scores = scores - scores.max(dim=-1, keepdim=True).values
        probs = torch.softmax(scores, dim=-1)
        out[s:e] = torch.einsum("hls,shd->lhd", probs, v)
    return out


class MultiheadSelfAttention(nn.Module):
    def __init__(self, num_channels: int, head_dim: int):
        super().__init__()
        assert num_channels % head_dim == 0
        self.num_heads = num_channels // head_dim
        self.head_dim = head_dim

        self.to_query = nn.Linear(num_channels, num_channels, bias=True)
        self.to_key = nn.Linear(num_channels, num_channels, bias=True)
        self.to_value = nn.Linear(num_channels, num_channels, bias=True)
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)
        self.out_layer = nn.Linear(num_channels, num_channels, bias=True)

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        shape = x.shape[:-1]
        q = self.to_query(x)
        k = self.to_key(x)
        v = self.to_value(x)

        q = self.query_norm(q.reshape(*shape, self.num_heads, -1)).type_as(q)
        k = self.key_norm(k.reshape(*shape, self.num_heads, -1)).type_as(k)
        v = v.reshape(*shape, self.num_heads, -1)
        qkv = torch.stack([q, k, v], dim=-3)

        use_flash = (
            _HAS_FLASH
            and x.is_cuda
            and qkv.dtype in (torch.float16, torch.bfloat16)
            and torch.cuda.get_device_capability(x.device)[0] >= 8  # Ampere+
        )
        if use_flash:
            max_seqlen = torch.diff(cu_seqlens).max().item()
            out = _fa_varlen_qkvpacked(qkv, cu_seqlens, max_seqlen)
        else:
            out = _varlen_attn_reference(qkv, cu_seqlens)

        out = out.flatten(-2, -1)
        return self.out_layer(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, ff_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(dim, ff_dim, bias=False)
        self.activation = nn.GELU()
        self.out_layer = nn.Linear(ff_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.activation(self.in_layer(x)))


class ModelBlock(nn.Module):
    def __init__(self, dim: int, ff_dim: int, head_dim: int):
        super().__init__()
        self.attention_norm = nn.RMSNorm(dim)
        self.attention = MultiheadSelfAttention(dim, head_dim)
        self.feed_forward_norm = nn.RMSNorm(dim)
        self.feed_forward = FeedForward(dim, ff_dim)

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), cu_seqlens)
        x = x + self.feed_forward(self.feed_forward_norm(x))
        return x


class Model(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, ff_dim: int, num_layers: int, head_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ModelBlock(hidden_dim, ff_dim, head_dim) for _ in range(num_layers)]
        )
        self.out_layer = nn.Linear(hidden_dim, in_dim)

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        x = self.in_layer(x)
        for blk in self.blocks:
            x = blk(x, cu_seqlens)
        return self.out_layer(x)
