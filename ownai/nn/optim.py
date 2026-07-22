"""Adam optimizer and a cosine learning-rate schedule, written from scratch."""
from __future__ import annotations

import math

from ownai.backend import xp


class Adam:
    """Adam (Kingma & Ba, 2014) with bias correction and optional weight decay."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [xp.zeros_like(p.data) for p in self.params]
        self.v = [xp.zeros_like(p.data) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.t += 1
        bc1 = 1.0 - self.b1**self.t
        bc2 = 1.0 - self.b2**self.t
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay:
                g = g + self.weight_decay * p.data
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            m_hat = self.m[i] / bc1
            v_hat = self.v[i] / bc2
            p.data = p.data - self.lr * m_hat / (xp.sqrt(v_hat) + self.eps)


def cosine_lr(step: int, *, warmup: int, max_steps: int, base_lr: float, min_lr: float = 0.0) -> float:
    """Linear warmup then cosine decay to min_lr."""
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
