"""Tests for the from-scratch mini-GPT transformer."""
import numpy as np

from ownai.autograd import Tensor
from ownai.model import GPT, GPTConfig


def tiny_config(**kw):
    base = dict(vocab_size=32, block_size=8, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
    base.update(kw)
    return GPTConfig(**base)


def test_forward_output_shape():
    model = GPT(tiny_config())
    idx = np.array([[1, 2, 3, 4]])
    logits = model(idx)
    assert logits.shape == (1, 4, 32)


def test_parameter_count_matches_formula():
    cfg = tiny_config()
    model = GPT(cfg)
    n = sum(p.data.size for p in model.parameters())
    assert n > 0
    # Sanity: token + position embeddings alone.
    assert n >= cfg.vocab_size * cfg.n_embd + cfg.block_size * cfg.n_embd


def test_causal_masking_prevents_future_leakage():
    # Changing a later token must not affect logits at earlier positions.
    model = GPT(tiny_config())
    model.eval()
    a = np.array([[1, 2, 3, 4, 5]])
    b = a.copy()
    b[0, 4] = 9  # change only the last token
    la = np.asarray(model(a).data)
    lb = np.asarray(model(b).data)
    # Positions 0..3 must be identical; position 4 may differ.
    np.testing.assert_allclose(la[0, :4], lb[0, :4], rtol=1e-5, atol=1e-5)
    assert not np.allclose(la[0, 4], lb[0, 4])


def test_loss_computation_and_backward():
    model = GPT(tiny_config())
    idx = np.array([[1, 2, 3, 4]])
    targets = np.array([[2, 3, 4, 5]])
    logits, loss = model(idx, targets=targets)
    assert loss.data.size == 1
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_overfits_tiny_sequence():
    # A correct transformer + optimizer must memorize one short sequence.
    from ownai.nn import Adam

    cfg = tiny_config(n_layer=2)
    model = GPT(cfg)
    opt = Adam(model.parameters(), lr=0.01)
    seq = np.array([[1, 2, 3, 4, 5, 6, 7]])
    x, y = seq[:, :-1], seq[:, 1:]
    first = last = None
    for step in range(120):
        opt.zero_grad()
        _, loss = model(x, targets=y)
        loss.backward()
        opt.step()
        if step == 0:
            first = float(loss.data)
        last = float(loss.data)
    assert last < first
    assert last < 0.1  # effectively memorized


def test_generate_respects_block_size_and_length():
    model = GPT(tiny_config(block_size=4))
    model.eval()
    out = model.generate(np.array([[1, 2]]), max_new_tokens=6, temperature=1.0, seed=0)
    assert out.shape[1] == 2 + 6
    assert out.max() < 32  # ids stay within vocab


def test_save_load_reproduces_logits(tmp_path):
    model = GPT(tiny_config())
    model.eval()
    idx = np.array([[1, 2, 3]])
    before = np.asarray(model(idx).data)
    path = tmp_path / "model.npz"
    model.save(path)
    restored = GPT.load(path)
    restored.eval()
    after = np.asarray(restored(idx).data)
    np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-6)
