"""Word2Vec (skip-gram with negative sampling), implemented from scratch.

Learns a dense vector per word such that words appearing in similar contexts
end up with similar vectors. Trained with plain numpy SGD — this is the
semantic half of the hybrid retriever, complementing BM25's lexical matching.
"""
from __future__ import annotations

from collections import Counter

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class Word2Vec:
    def __init__(self, dim: int = 64, window: int = 3, negative: int = 5, min_count: int = 1, seed: int = 0):
        self.dim = dim
        self.window = window
        self.negative = negative
        self.min_count = min_count
        self.rng = np.random.default_rng(seed)
        self.vocab: dict[str, int] = {}
        self.W_in = None   # target/word embeddings (the ones we keep)
        self.W_out = None  # context embeddings

    # ----------------------------------------------------------------- build
    def _build_vocab(self, sentences):
        counts = Counter(w for s in sentences for w in s)
        self.vocab = {w: i for i, (w, c) in enumerate(counts.items()) if c >= self.min_count}
        vsize = len(self.vocab)
        # Unigram^0.75 negative sampling distribution.
        freqs = np.zeros(vsize)
        for w, i in self.vocab.items():
            freqs[i] = counts[w]
        p = freqs**0.75
        self.neg_p = p / p.sum()
        self.W_in = (self.rng.standard_normal((vsize, self.dim)) * 0.01).astype(np.float64)
        self.W_out = np.zeros((vsize, self.dim), dtype=np.float64)

    # ----------------------------------------------------------------- train
    def train(self, sentences, epochs: int = 5, lr: float = 0.05):
        self._build_vocab(sentences)
        indexed = [[self.vocab[w] for w in s if w in self.vocab] for s in sentences]
        vsize = len(self.vocab)
        for _ in range(epochs):
            self.rng.shuffle(indexed)
            for sent in indexed:
                for pos, target in enumerate(sent):
                    lo = max(0, pos - self.window)
                    hi = min(len(sent), pos + self.window + 1)
                    for ctx_pos in range(lo, hi):
                        if ctx_pos == pos:
                            continue
                        context = sent[ctx_pos]
                        self._update(target, context, vsize, lr)

    def _update(self, target: int, context: int, vsize: int, lr: float):
        # Positive sample + `negative` sampled noise words.
        negs = self.rng.choice(vsize, size=self.negative, p=self.neg_p)
        samples = np.concatenate(([context], negs))
        labels = np.zeros(len(samples))
        labels[0] = 1.0

        v_in = self.W_in[target]                 # (dim,)
        v_out = self.W_out[samples]              # (k+1, dim)
        score = _sigmoid(v_out @ v_in)           # (k+1,)
        grad = (score - labels) * lr             # (k+1,)

        # Gradient w.r.t input vector, then update output vectors.
        grad_in = grad @ v_out                   # (dim,)
        self.W_out[samples] -= np.outer(grad, v_in)
        self.W_in[target] -= grad_in

    # ------------------------------------------------------------- inference
    def vector(self, word: str):
        idx = self.vocab.get(word)
        return None if idx is None else self.W_in[idx]

    def similarity(self, a: str, b: str) -> float:
        va, vb = self.vector(a), self.vector(b)
        if va is None or vb is None:
            return 0.0
        denom = (np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-9
        return float(va @ vb / denom)

    def embed_tokens(self, tokens: list[str]) -> np.ndarray:
        """Mean of known word vectors — a simple sentence embedding."""
        vecs = [self.W_in[self.vocab[t]] for t in tokens if t in self.vocab]
        if not vecs:
            return np.zeros(self.dim)
        return np.mean(vecs, axis=0)
