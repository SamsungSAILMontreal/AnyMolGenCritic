# Modified from nanogpt github repo

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Type, Union, Tuple
from torch.nn import functional as F
import math
from dataclasses import dataclass

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Mistral
class MistralRotaryEmbedding(nn.Module):
    def __init__(self, dim, seq_len=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.seq_len = seq_len
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=seq_len, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

        self.position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)
        self.position_ids = self.position_ids.unsqueeze(0).view(-1, seq_len)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq) # L x D/2
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1) # L x D
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, q, k):#, seq_len=None):
        seq_len = q.shape[2]

        cos = self.cos_cached.to(dtype=q.dtype)
        cos = cos[self.position_ids[:, 0:seq_len]].unsqueeze(1)
        sin = self.sin_cached.to(dtype=q.dtype)
        sin = sin[self.position_ids[:, 0:seq_len]].unsqueeze(1)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

# From https://huggingface.co/mosaicml/mpt-7b/blob/main/norm.py
class RMSNorm(torch.nn.Module):

    def __init__(self, ndim, eps: float=1e-05, weight: bool = True, dtype: Optional[torch.dtype]=None, device: Optional[torch.device]=None):
        super().__init__()
        self.eps = eps
        if weight:
            self.weight = torch.nn.Parameter(torch.ones(ndim))
        else:
            self.register_parameter('weight', None)
        self.use_weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.use_weight:
            output = output * self.weight
        return output

class CausalSelfAttention(nn.Module):

    def __init__(self, config, rotary_emb=None):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attnq = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_attnk = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_attnv = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.rotary_emb = rotary_emb

    def forward(self, x, mask):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        q  = self.c_attnq(x)
        k  = self.c_attnk(x)
        v  = self.c_attnv(x)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # rotary embeddings
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(q, k)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0, is_causal=mask is None)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.expand_scale = config.expand_scale
        self.c_fc    = nn.Linear(config.n_embd, int(self.expand_scale * config.n_embd), bias=config.bias)
        self.c_proj  = nn.Linear(int(self.expand_scale * config.n_embd), config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.swiglu = config.swiglu
        if self.swiglu:
            self.act = nn.SiLU()
            self.c_fc2 = nn.Linear(config.n_embd, int(self.expand_scale * config.n_embd), bias=config.bias)
        else:
            self.act = nn.GELU()
        self.n_head = config.n_head

    def forward(self, x):
        y = self.act(self.c_fc(x))
        if self.swiglu:
            y2 = self.c_fc2(x)
            y = y*y2
        z = self.dropout(self.c_proj(y))
        return z

class Block(nn.Module):

    def __init__(self, config, rotary_emb):
        super().__init__()

        if config.rmsnorm:
            self.ln_1 = RMSNorm(config.n_embd, weight=True)
        else:
            self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)

        self.attn = CausalSelfAttention(config, rotary_emb)

        if config.rmsnorm:
            self.ln_2 = RMSNorm(config.n_embd, weight=True)
        else:
            self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)

        self.mlp = MLP(config)
        self.config = config

    def forward(self, x, mask, emb):
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class TransformerConfig:
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    rotary: bool = False # rotary embedding
    rmsnorm: bool = False # RMSNorm instead of LayerNorm
    swiglu: bool = False # SwiGLU instead of GELU
    expand_scale: float = 3.5

class Transformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.block_size is not None
        self.config = config

        if config.rotary:
            assert config.n_embd % (2*config.n_head) == 0 # n_emb must be divisible by n_heads multiplied by 2
            self.rotary_emb = MistralRotaryEmbedding(
                config.n_embd // config.n_head,
                seq_len=config.block_size,
                base=10000,
            )
        else:
            self.rotary_emb = None

        self.transformer = nn.ModuleDict(dict(
            h = nn.ModuleList([Block(config, self.rotary_emb) for i in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd, weight=True) if config.rmsnorm else LayerNorm(config.n_embd, bias=config.bias),
        ))

    def forward(self, x, mask=None, emb_cond=None):

        # forward the GPT model itself
        for block in self.transformer.h:
            x = block(x, mask, emb_cond)
        x = self.transformer.ln_f(x)

        return x