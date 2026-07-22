"""Tests for neural-network layers, loss and optimizer built on the autograd."""
import numpy as np

from ownai.autograd import Tensor
from ownai.nn import (
    Adam,
    CrossEntropyLoss,
    Dropout,
    Embedding,
    LayerNorm,
    Linear,
    Module,
)

RNG = np.random.default_rng(1)


def numeric_param_grad(module, loss_fn, param, eps=1e-5):
    """Finite-difference gradient of a scalar loss w.r.t. one parameter array."""
    grad = np.zeros_like(param.data)
    flat = param.data.reshape(-1)
    gflat = grad.reshape(-1)
    for i in range(flat.size):
        orig = float(flat[i])
        flat[i] = orig + eps
        lp = float(loss_fn().data)
        flat[i] = orig - eps
        lm = float(loss_fn().data)
        flat[i] = orig
        gflat[i] = (lp - lm) / (2 * eps)
    return grad


def test_linear_shapes_and_params():
    layer = Linear(4, 3)
    x = Tensor(RNG.standard_normal((5, 4)))
    y = layer(x)
    assert y.shape == (5, 3)
    names = {id(p) for p in layer.parameters()}
    assert len(names) == 2  # weight + bias


def test_linear_gradients_match_numeric():
    layer = Linear(3, 2)
    x = Tensor(RNG.standard_normal((4, 3)).astype(np.float64))
    for p in layer.parameters():
        p.data = p.data.astype(np.float64)

    def loss_fn():
        return (layer(x) ** 2).sum()

    loss = loss_fn()
    loss.backward()
    for p in layer.parameters():
        expected = numeric_param_grad(layer, loss_fn, p)
        np.testing.assert_allclose(np.asarray(p.grad), expected, rtol=1e-4, atol=1e-4)


def test_embedding_lookup_and_grad():
    emb = Embedding(6, 3)
    emb.weight.data = emb.weight.data.astype(np.float64)
    idx = np.array([[0, 2], [2, 5]])

    def loss_fn():
        return (emb(idx) ** 2).sum()

    loss = loss_fn()
    assert loss  # scalar
    out = emb(idx)
    assert out.shape == (2, 2, 3)
    loss.backward()
    expected = numeric_param_grad(emb, loss_fn, emb.weight)
    np.testing.assert_allclose(np.asarray(emb.weight.grad), expected, rtol=1e-4, atol=1e-4)


def test_layernorm_normalizes_and_grad():
    ln = LayerNorm(4)
    for p in ln.parameters():
        p.data = p.data.astype(np.float64)
    x = Tensor(RNG.standard_normal((3, 4)).astype(np.float64))
    out = ln(x)
    # With default gamma=1, beta=0 each row should be ~zero-mean, unit-var.
    row0 = np.asarray(out.data)[0]
    assert abs(row0.mean()) < 1e-6

    def loss_fn():
        return (ln(x) ** 2).sum()

    loss = loss_fn()
    loss.backward()
    for p in ln.parameters():
        expected = numeric_param_grad(ln, loss_fn, p)
        np.testing.assert_allclose(np.asarray(p.grad), expected, rtol=1e-4, atol=1e-4)


def test_cross_entropy_matches_manual():
    logits = Tensor(RNG.standard_normal((3, 5)).astype(np.float64), requires_grad=True)
    targets = np.array([1, 0, 4])
    loss = CrossEntropyLoss()(logits, targets)
    # Manual reference: -mean(log_softmax[row, target]).
    z = np.asarray(logits.data)
    logp = z - np.log(np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)) - z.max(1, keepdims=True)
    ref = -logp[np.arange(3), targets].mean()
    np.testing.assert_allclose(float(loss.data), ref, rtol=1e-6)
    loss.backward()
    assert logits.grad is not None


def test_dropout_eval_is_identity():
    drop = Dropout(0.5)
    drop.eval()
    x = Tensor(np.ones((4, 4)))
    out = drop(x)
    np.testing.assert_allclose(np.asarray(out.data), np.ones((4, 4)))


def test_adam_solves_linear_regression():
    # A solvable problem: target is a true linear function of x, so a correct
    # optimizer must drive the loss close to zero.
    layer = Linear(3, 1)
    opt = Adam(layer.parameters(), lr=0.05)
    x = Tensor(RNG.standard_normal((32, 3)))
    true_w = np.array([[1.5], [-2.0], [0.5]])
    target = Tensor(np.asarray(x.data) @ true_w + 0.3)
    first, last = None, None
    for step in range(400):
        opt.zero_grad()
        loss = ((layer(x) - target) ** 2).mean()
        loss.backward()
        opt.step()
        if step == 0:
            first = float(loss.data)
        last = float(loss.data)
    assert last < 1e-3
    assert last < first * 0.01


def test_module_train_eval_toggles_children():
    class Net(Module):
        def __init__(self):
            super().__init__()
            self.drop = Dropout(0.5)

        def forward(self, x):
            return self.drop(x)

    net = Net()
    net.eval()
    assert net.drop.training is False
    net.train()
    assert net.drop.training is True
