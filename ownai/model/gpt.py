"""A decoder-only transformer (GPT), built entirely on our own autograd.

Architecture (per GPT-2, scaled down):
    token embedding + learned positional embedding
    -> N x [ LayerNorm -> causal multi-head self-attention -> residual
             LayerNorm -> MLP (4x) with ReLU -> residual ]
    -> final LayerNorm -> linear head to vocab logits

Every op (attention scores, causal mask, softmax, residuals) flows through the
Tensor autograd, so training uses our hand-written backprop end to end.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ownai.autograd import Tensor
from ownai.backend import asnumpy, xp
from ownai.model.config import GPTConfig
from ownai.nn import CrossEntropyLoss, Dropout, LayerNorm, Linear, Module, ModuleList


class CausalSelfAttention(Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = Linear(cfg.n_embd, cfg.n_embd)
        self.attn_drop = Dropout(cfg.dropout)
        self.resid_drop = Dropout(cfg.dropout)
        # Lower-triangular mask: 0 where allowed, -inf where future.
        mask = np.triu(np.full((cfg.block_size, cfg.block_size), -1e9, dtype=np.float32), k=1)
        self._mask = xp.asarray(mask)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)                                  # (B,T,3C)
        q = qkv[:, :, :C]
        k = qkv[:, :, C : 2 * C]
        v = qkv[:, :, 2 * C :]

        def split_heads(t):
            return t.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)   # (B,nh,T,hd)

        att = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / (self.head_dim**0.5))  # (B,nh,T,T)
        att = att + Tensor(self._mask[:T, :T])
        att = att.softmax(axis=-1)
        att = self.attn_drop(att)
        out = att @ v                                       # (B,nh,T,hd)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)    # merge heads
        return self.resid_drop(self.proj(out))


class MLP(Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.proj(self.fc(x).relu()))


class Block(Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        from ownai.nn import Embedding

        self.wte = Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = Embedding(cfg.block_size, cfg.n_embd)
        self.drop = Dropout(cfg.dropout)
        self.blocks = ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.n_embd)
        self.head = Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        idx = xp.asarray(idx)
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")
        pos = xp.arange(T)
        x = self.wte(idx) + self.wpe(pos)          # (B,T,C), pos broadcasts over batch
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)                       # (B,T,vocab)
        if targets is None:
            return logits
        V = self.cfg.vocab_size
        loss = CrossEntropyLoss()(logits.reshape(B * T, V), xp.asarray(targets).reshape(B * T))
        return logits, loss

    # ------------------------------------------------------------- generation
    def generate(
        self,
        idx,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k=None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        seed=0,
    ):
        """Autoregressively sample continuations, NumPy-only and seed-deterministic.

        Sampling knobs, applied in this order to each step's logits:
          1. repetition_penalty (CTRL, arXiv:1909.05858): for every token id
             already present in that row's sequence, divide its logit by the
             penalty when positive, else multiply -- both push the id toward
             lower probability, discouraging loops. penalty=1.0 is a no-op.
          2. temperature: flatten (>1) or sharpen (<1) the distribution.
          3. top_k: keep only the k highest logits.
          4. top_p (nucleus): keep the smallest set of highest-probability
             tokens whose cumulative probability reaches top_p, then renormalize.
             Applied after top_k when both are given; top_p=1.0 filters nothing.
        """
        rng = np.random.default_rng(seed)
        idx = np.asarray(idx)
        for _ in range(max_new_tokens):
            cond = idx[:, -self.cfg.block_size :]
            logits = asnumpy(self(cond).data)[:, -1, :]     # (B,vocab)

            # (1) Repetition penalty on raw logits, per row (each row has its
            #     own history of already-emitted / prompt token ids).
            if repetition_penalty != 1.0:
                for b in range(logits.shape[0]):
                    seen = np.unique(idx[b])                 # ids present in this row
                    row = logits[b, seen]
                    logits[b, seen] = np.where(
                        row > 0, row / repetition_penalty, row * repetition_penalty
                    )

            # (2) Temperature.
            logits = logits / max(temperature, 1e-6)

            # (3) Top-k: mask everything below the k-th largest logit.
            if top_k is not None:
                kth = np.sort(logits, axis=-1)[:, -top_k][:, None]
                logits = np.where(logits < kth, -np.inf, logits)

            probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs /= probs.sum(axis=-1, keepdims=True)

            # (4) Top-p / nucleus. top_p=1.0 keeps every token (the smallest set
            #     reaching cumulative prob 1.0 is the whole vocab), so we skip the
            #     work and leave probs untouched -- identical to the no-top_p path.
            if top_p is not None and top_p < 1.0:
                order = np.argsort(probs, axis=-1)[:, ::-1]         # descending
                sorted_probs = np.take_along_axis(probs, order, axis=-1)
                cum = np.cumsum(sorted_probs, axis=-1)
                # Drop tokens once cumulative prob has passed top_p, but always
                # keep the first token that crosses the threshold (shift by one).
                remove = cum > top_p
                remove[:, 1:] = remove[:, :-1]
                remove[:, 0] = False
                sorted_probs = np.where(remove, 0.0, sorted_probs)
                # Scatter the kept mass back to the original vocab order.
                keep = np.zeros_like(probs)
                np.put_along_axis(keep, order, sorted_probs, axis=-1)
                probs = keep / keep.sum(axis=-1, keepdims=True)

            next_ids = np.array([rng.choice(probs.shape[1], p=p) for p in probs])
            idx = np.concatenate([idx, next_ids[:, None]], axis=1)
        return idx

    # ------------------------------------------------------------- persistence
    def save(self, path) -> None:
        arrays = {f"param::{name}": asnumpy(p.data) for name, p in self.named_parameters()}
        for key, val in self.cfg.to_dict().items():
            arrays[f"cfg::{key}"] = np.array(val)
        np.savez(Path(path), **arrays)

    @classmethod
    def load(cls, path) -> "GPT":
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        data = np.load(path, allow_pickle=False)
        cfg_kwargs = {}
        for key in data.files:
            if key.startswith("cfg::"):
                raw = data[key]
                cfg_kwargs[key[5:]] = raw.item()
        cfg = GPTConfig(
            vocab_size=int(cfg_kwargs["vocab_size"]),
            block_size=int(cfg_kwargs["block_size"]),
            n_layer=int(cfg_kwargs["n_layer"]),
            n_head=int(cfg_kwargs["n_head"]),
            n_embd=int(cfg_kwargs["n_embd"]),
            dropout=float(cfg_kwargs["dropout"]),
        )
        model = cls(cfg)
        state = {k[7:]: data[k] for k in data.files if k.startswith("param::")}
        model.load_state_dict(state)
        return model
