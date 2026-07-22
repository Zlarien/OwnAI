"""Tests for the language-model training utilities."""
import numpy as np

from ownai.model import GPT, GPTConfig
from ownai.model.train import LMDataset, evaluate_perplexity, train_lm


def test_lm_dataset_batches_shapes():
    tokens = list(range(100))
    ds = LMDataset(tokens, block_size=8, seed=0)
    x, y = ds.batch(batch_size=4)
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    # y is x shifted by one position.
    # Reconstruct: for each row, y[t] should equal token after x[t].
    assert (y[:, :-1] == x[:, 1:]).all()


def test_lm_dataset_raises_when_too_short():
    import pytest

    with pytest.raises(ValueError):
        LMDataset(list(range(4)), block_size=8)


def test_train_lm_reduces_loss_on_repeating_pattern():
    # A periodic token stream is highly predictable; loss must drop.
    pattern = [1, 2, 3, 4, 5, 6, 7, 8] * 40
    cfg = GPTConfig(vocab_size=16, block_size=8, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    model = GPT(cfg)
    history = train_lm(
        model, pattern, steps=60, batch_size=8, lr=0.01, seed=0, log_every=1000
    )
    assert history["loss"][-1] < history["loss"][0]
    assert history["loss"][-1] < 1.0


def test_evaluate_perplexity_is_finite():
    pattern = [1, 2, 3, 4, 5, 6, 7, 8] * 20
    cfg = GPTConfig(vocab_size=16, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
    model = GPT(cfg)
    ppl = evaluate_perplexity(model, pattern, block_size=8, batch_size=4, seed=0, batches=3)
    assert np.isfinite(ppl)
    assert ppl > 0
