"""Tests for the extended GPT.generate sampling (top_p nucleus + repetition penalty).

These exercise the from-scratch, NumPy-only sampler: everything must stay
deterministic for a fixed seed and never leave the vocabulary.
"""
import numpy as np

from ownai.model import GPT, GPTConfig


def tiny_config(**kw):
    # Same shape as tests/test_model.py so behaviour is comparable.
    base = dict(vocab_size=32, block_size=8, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
    base.update(kw)
    return GPTConfig(**base)


def _count_repeats(row):
    """Number of tokens in a 1-D sequence that already appeared earlier."""
    return len(row) - len(np.unique(row))


def test_top_p_one_matches_no_top_p():
    # top_p=1.0 keeps the whole distribution, so with the same seed it must
    # reproduce the plain (no top_p) sampling exactly -- it filters nothing.
    np.random.seed(0)  # deterministic weights, independent of test order
    model = GPT(tiny_config())
    model.eval()
    prompt = np.array([[1, 2, 3]])

    baseline = model.generate(prompt, max_new_tokens=10, temperature=1.0, seed=7)
    nucleus = model.generate(prompt, max_new_tokens=10, temperature=1.0, top_p=1.0, seed=7)

    np.testing.assert_array_equal(baseline, nucleus)


def test_repetition_penalty_runs_and_does_not_increase_repeats():
    # A penalty > 1 must keep output in-vocab, change the sampled sequence, and
    # not produce MORE repeated tokens than the un-penalised baseline.
    np.random.seed(0)  # deterministic weights, independent of test order
    model = GPT(tiny_config())
    model.eval()
    prompt = np.array([[1, 2, 3, 4]])
    n_new = 12

    base = model.generate(prompt, max_new_tokens=n_new, temperature=1.0, seed=3)
    pen = model.generate(
        prompt, max_new_tokens=n_new, temperature=1.0, repetition_penalty=2.0, seed=3
    )

    assert pen.shape == base.shape
    assert pen.min() >= 0 and pen.max() < 32          # stays within vocab
    assert not np.array_equal(base, pen)              # the penalty has an effect

    base_reps = _count_repeats(base[0])
    pen_reps = _count_repeats(pen[0])
    assert pen_reps <= base_reps                       # penalty discourages loops


def test_output_shape_and_vocab_bounds():
    # Shape is (B, T + max_new_tokens) and every id is a valid vocab index,
    # exercising top_k + top_p + repetition_penalty together.
    np.random.seed(0)  # deterministic weights, independent of test order
    cfg = tiny_config()
    model = GPT(cfg)
    model.eval()
    prompt = np.array([[1, 2], [3, 4]])                # B=2, T=2
    n_new = 5

    out = model.generate(
        prompt,
        max_new_tokens=n_new,
        temperature=0.8,
        top_k=8,
        top_p=0.9,
        repetition_penalty=1.2,
        seed=0,
    )

    assert out.shape == (2, 2 + n_new)
    assert out.min() >= 0
    assert out.max() < cfg.vocab_size
