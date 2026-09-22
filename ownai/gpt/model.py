"""OwnGPT: the scaled-up decoder-only transformer, written from scratch on PyTorch.

The NumPy GPT in ``ownai/model`` proves every gradient by hand; it cannot train
on billions of tokens. This one keeps the same idea but uses PyTorch tensors and
kernels so it runs on a GPU. Every layer below is written here, nothing comes
from a model zoo:

    token embedding (tied with the output head)
    -> N x [ RMSNorm -> causal self-attention with RoPE -> residual
             RMSNorm -> SwiGLU MLP -> residual ]
    -> RMSNorm -> logits
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 32768  # our BPE tokenizer (see tokenizer.py), a multiple of 64
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# Sizes that match the hardware they are meant for. Parameter counts assume the
# 32k vocabulary and include the (tied) embedding table.
PRESETS = {
    "tiny": dict(n_layer=2, n_head=2, n_embd=64, block_size=64),        # ~2M, CPU smoke tests
    "small": dict(n_layer=6, n_head=6, n_embd=384, block_size=512),     # ~23M, quick GPU runs
    "t4": dict(n_layer=8, n_head=8, n_embd=512, block_size=512),        # ~42M, free T4 (Colab / Kaggle)
    "base": dict(n_layer=12, n_head=12, n_embd=768, block_size=1024),   # ~110M, A100 / H100
    "medium": dict(n_layer=24, n_head=16, n_embd=1024, block_size=1024),  # ~330M, rented H100s
}


def config_from_preset(name: str, **overrides) -> GPTConfig:
    return GPTConfig(**{**PRESETS[name], **overrides})


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def rope_tables(seq_len: int, head_dim: int, device, base: float = 10000.0):
    """Cos/sin tables for rotary position embeddings, shape (seq_len, head_dim/2)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    angles = torch.outer(torch.arange(seq_len, device=device).float(), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotate each (first half, second half) pair of a head by its position angle."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.proj.residual = True
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, cache=None, layer: int = 0):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:  # KV cache: keep past keys/values, only compute the new tokens
            if cache[layer] is not None:
                k = torch.cat([cache[layer][0], k], dim=2)
                v = torch.cat([cache[layer][1], v], dim=2)
            cache[layer] = (k, v)
        # Full causal mask on a fresh sequence; a single decoded token sees everything cached.
        assert T == k.shape[2] or T == 1, "cached decoding goes one token at a time"
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=T > 1, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = 64 * math.ceil(8 * cfg.n_embd / 3 / 64)  # same params as a 4x GELU MLP
        self.gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.down.residual = True
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None, layer: int = 0):
        x = x + self.attn(self.norm1(x), cos, sin, cache, layer)
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.wte.weight  # weight tying
        cos, sin = rope_tables(cfg.block_size, cfg.n_embd // cfg.n_head, "cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            std = 0.02
            if getattr(m, "residual", False):  # GPT-2: damp the residual stream growth
                std /= math.sqrt(2 * self.cfg.n_layer)
            nn.init.normal_(m.weight, mean=0.0, std=std)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, cache=None, pos: int = 0):
        """Logits for every position; loss if ``targets`` is given (-1 = ignored).

        ``cache`` (a list with one slot per layer) and ``pos`` are only used for
        fast generation, where tokens arrive one at a time.
        """
        T = idx.shape[1]
        assert pos + T <= self.cfg.block_size, f"sequence {pos + T} > block_size {self.cfg.block_size}"
        cos, sin = self.rope_cos[pos:pos + T], self.rope_sin[pos:pos + T]
        x = self.drop(self.wte(idx))
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, cache, i)
        x = self.norm(x)
        if targets is None:
            return self.head(x[:, [-1], :]), None
        logits = self.head(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    def configure_optimizer(self, lr: float, weight_decay: float, betas=(0.9, 0.95)):
        decay = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
        fused = next(self.parameters()).is_cuda
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.8, top_k: int | None = 50,
                 top_p: float | None = 0.95, stop_token: int | None = None, generator=None):
        """``idx`` (1, T) followed by up to ``max_new_tokens`` sampled tokens."""
        new = list(self.stream(idx, max_new_tokens, temperature, top_k, top_p, stop_token, generator))
        return torch.cat([idx, torch.tensor([new], dtype=idx.dtype, device=idx.device)], dim=1)

    @torch.no_grad()
    def stream(self, idx, max_new_tokens: int, temperature: float = 0.8, top_k: int | None = 50,
               top_p: float | None = 0.95, stop_token: int | None = None, generator=None,
               repetition_penalty: float = 0.0, penalty_window: int = 128):
        """Yield sampled token ids one by one (the stop token included), with a KV cache.

        ``repetition_penalty`` subtracts that much from the score of a token per
        time it already appeared in the last ``penalty_window`` positions, which
        breaks the loops a small model falls into ("pratiquer, pratiquer...").
        """
        # The prompt keeps priority: generation gets whatever room is left in the
        # window, but at least a quarter of it (the prompt's oldest tokens go first).
        block = self.cfg.block_size
        max_new = min(max_new_tokens, max(block - idx.shape[1], block // 4))
        idx = idx[:, -(block - max_new):]
        cache = [None] * self.cfg.n_layer
        logits, _ = self(idx, cache=cache)
        pos = idx.shape[1]
        seen = idx[0, -penalty_window:].tolist()
        for step in range(max_new):
            nxt = self._sample(logits[:, -1, :].float(), temperature, top_k, top_p, generator,
                               repetition_penalty, seen)
            token = int(nxt[0, 0])
            seen.append(token)
            del seen[:-penalty_window]
            yield token
            if token == stop_token or step == max_new - 1:
                return
            logits, _ = self(nxt, cache=cache, pos=pos)
            pos += 1

    @staticmethod
    def _sample(logits, temperature, top_k, top_p, generator, repetition_penalty=0.0, seen=()):
        if repetition_penalty and len(seen):
            ids = torch.tensor(seen, device=logits.device)
            counts = torch.zeros_like(logits[0]).index_add_(0, ids, torch.ones_like(ids, dtype=logits.dtype))
            logits -= repetition_penalty * counts  # the more a token was used, the harder it gets
        if temperature <= 0:
            return logits.argmax(-1, keepdim=True)
        logits = logits / temperature
        if top_k:
            kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
            logits[logits < kth] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        if top_p is not None and top_p < 1.0:
            sorted_p, order = probs.sort(descending=True)
            sorted_p[sorted_p.cumsum(-1) - sorted_p > top_p] = 0.0
            probs = torch.zeros_like(probs).scatter(-1, order, sorted_p)
            probs = probs / probs.sum(-1, keepdim=True)
        return torch.multinomial(probs, 1, generator=generator)
