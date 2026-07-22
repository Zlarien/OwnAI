"""Language-model training loop, dataset batching and perplexity evaluation."""
from __future__ import annotations

import math

import numpy as np

from ownai.backend import xp
from ownai.model.gpt import GPT
from ownai.nn import Adam
from ownai.nn.optim import cosine_lr


class LMDataset:
    """Contiguous token stream sliced into random (x, y=next-token) windows."""

    def __init__(self, tokens, block_size: int, seed: int = 0):
        if len(tokens) <= block_size:
            raise ValueError("token stream shorter than block_size + 1")
        self.tokens = np.asarray(tokens, dtype=np.int64)
        self.block_size = block_size
        self.rng = np.random.default_rng(seed)

    def batch(self, batch_size: int):
        max_start = len(self.tokens) - self.block_size - 1
        starts = self.rng.integers(0, max_start + 1, size=batch_size)
        x = np.stack([self.tokens[s : s + self.block_size] for s in starts])
        y = np.stack([self.tokens[s + 1 : s + 1 + self.block_size] for s in starts])
        return x, y


def train_lm(
    model: GPT,
    tokens,
    *,
    steps: int,
    batch_size: int = 16,
    lr: float = 3e-4,
    warmup: int = 0,
    min_lr: float | None = None,
    weight_decay: float = 0.0,
    seed: int = 0,
    log_every: int = 50,
    on_log=None,
) -> dict:
    """Train a GPT for `steps` optimizer steps. Returns loss/lr history."""
    model.train()
    ds = LMDataset(tokens, model.cfg.block_size, seed=seed)
    opt = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    min_lr = lr * 0.1 if min_lr is None else min_lr
    history = {"step": [], "loss": [], "lr": []}

    for step in range(steps):
        cur_lr = cosine_lr(step, warmup=warmup, max_steps=steps, base_lr=lr, min_lr=min_lr) if warmup else lr
        opt.lr = cur_lr
        x, y = ds.batch(batch_size)
        opt.zero_grad()
        _, loss = model(x, targets=y)
        loss.backward()
        opt.step()

        loss_val = float(loss.data)
        history["step"].append(step)
        history["loss"].append(loss_val)
        history["lr"].append(cur_lr)
        if step % log_every == 0 or step == steps - 1:
            msg = f"step {step:5d} | loss {loss_val:.4f} | lr {cur_lr:.2e}"
            (on_log or print)(msg)
    return history


def evaluate_perplexity(
    model: GPT, tokens, *, block_size: int, batch_size: int = 8, seed: int = 0, batches: int = 20
) -> float:
    """Average perplexity over sampled validation windows (lower is better)."""
    model.eval()
    ds = LMDataset(tokens, block_size, seed=seed)
    total = 0.0
    for _ in range(batches):
        x, y = ds.batch(batch_size)
        _, loss = model(x, targets=y)
        total += float(loss.data)
    return math.exp(total / batches)
