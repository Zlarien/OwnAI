"""Hybrid retriever = BM25 (lexical) + word2vec cosine (semantic).

Scores from each signal are min-max normalized to [0,1] and blended. Lexical
matching nails exact terms; semantic matching catches paraphrases the user
typed differently from the corpus. Together they beat either alone.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from ownai.data import Chunk
from ownai.retrieval.bm25 import BM25
from ownai.retrieval.tokenize import tokenize
from ownai.retrieval.word2vec import Word2Vec


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


class HybridRetriever:
    def __init__(self, embed_dim: int = 64, alpha: float = 0.5, seed: int = 0):
        self.embed_dim = embed_dim
        self.alpha = alpha  # weight on semantic score; (1-alpha) on lexical
        self.seed = seed
        self.chunks: list[Chunk] = []
        self.bm25: BM25 | None = None
        self.w2v: Word2Vec | None = None
        self.doc_vecs: np.ndarray | None = None

    def index(self, chunks: list[Chunk], w2v_epochs: int = 10, w2v_lr: float = 0.05) -> None:
        self.chunks = list(chunks)
        tokenized = [tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25(tokenized)
        self.w2v = Word2Vec(dim=self.embed_dim, seed=self.seed)
        self.w2v.train(tokenized, epochs=w2v_epochs, lr=w2v_lr)
        self.doc_vecs = np.array([self.w2v.embed_tokens(t) for t in tokenized])
        # Pre-normalize for cosine.
        norms = np.linalg.norm(self.doc_vecs, axis=1, keepdims=True) + 1e-9
        self.doc_vecs_unit = self.doc_vecs / norms

    def _semantic_scores(self, query_tokens: list[str]) -> np.ndarray:
        q = self.w2v.embed_tokens(query_tokens)
        qn = np.linalg.norm(q)
        if qn < 1e-9:
            return np.zeros(len(self.chunks))
        return self.doc_vecs_unit @ (q / qn)

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        q_tokens = tokenize(query)
        lexical = _minmax(self.bm25.scores(q_tokens))
        semantic = _minmax(self._semantic_scores(q_tokens))
        combined = self.alpha * semantic + (1 - self.alpha) * lexical
        order = np.argsort(-combined)[:top_k]
        return [(self.chunks[i], float(combined[i])) for i in order]

    # --------------------------------------------------------------- persistence
    def save(self, dir_path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        with open(dir_path / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")
        with open(dir_path / "state.pkl", "wb") as f:
            pickle.dump(
                {
                    "embed_dim": self.embed_dim,
                    "alpha": self.alpha,
                    "seed": self.seed,
                    "w2v": self.w2v,
                    "doc_vecs": self.doc_vecs,
                },
                f,
            )

    @classmethod
    def load(cls, dir_path) -> "HybridRetriever":
        # Safe: state.pkl is only ever the index this project produced locally
        # via save(); it is a build artifact, never a file from an untrusted
        # third party. Do not point this at downloaded/unverified pickles.
        dir_path = Path(dir_path)
        with open(dir_path / "state.pkl", "rb") as f:
            state = pickle.load(f)
        retr = cls(embed_dim=state["embed_dim"], alpha=state["alpha"], seed=state["seed"])
        chunks = []
        with open(dir_path / "chunks.jsonl", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(Chunk(**json.loads(line)))
        retr.chunks = chunks
        retr.w2v = state["w2v"]
        retr.doc_vecs = state["doc_vecs"]
        norms = np.linalg.norm(retr.doc_vecs, axis=1, keepdims=True) + 1e-9
        retr.doc_vecs_unit = retr.doc_vecs / norms
        retr.bm25 = BM25([tokenize(c.text) for c in chunks])
        return retr
