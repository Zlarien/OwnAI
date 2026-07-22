"""Concrete layers: Linear, Embedding, LayerNorm, Dropout."""
from __future__ import annotations

from ownai.autograd import Tensor
from ownai.backend import xp
from ownai.nn.module import Module, Parameter


def _kaiming(shape, fan_in):
    std = (2.0 / fan_in) ** 0.5
    return (xp.random.standard_normal(shape) * std).astype(xp.float32)


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = Parameter(_kaiming((in_features, out_features), in_features))
        self.bias = Parameter(xp.zeros(out_features, dtype=xp.float32)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class Embedding(Module):
    """A lookup table mapping integer ids to dense vectors."""

    def __init__(self, num_embeddings: int, dim: int):
        super().__init__()
        self.weight = Parameter((xp.random.standard_normal((num_embeddings, dim)) * 0.02).astype(xp.float32))

    def forward(self, ids) -> Tensor:
        # ids is a plain integer array; indexing routes gradients via scatter_add.
        return self.weight[xp.asarray(ids)]


class LayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = Parameter(xp.ones(dim, dtype=xp.float32))
        self.beta = Parameter(xp.zeros(dim, dtype=xp.float32))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(axis=-1, keepdims=True)
        centered = x - mean
        var = (centered * centered).mean(axis=-1, keepdims=True)
        std = (var + self.eps) ** 0.5
        normed = centered / std
        return normed * self.gamma + self.beta


class Dropout(Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        assert 0.0 <= p < 1.0
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        # Inverted dropout: scale at train time so eval is a plain identity.
        mask = (xp.random.random(x.shape) >= self.p).astype(x.dtype) / (1.0 - self.p)
        return x * Tensor(mask)
