"""BM25 ranking function, implemented from scratch.

BM25 scores how well a document matches a query using term frequency saturation
(k1) and document-length normalization (b), weighted by inverse document
frequency. It is the classic strong baseline for lexical retrieval.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np


class BM25:
    def __init__(self, tokenized_docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = tokenized_docs
        self.n = len(tokenized_docs)
        self.doc_len = np.array([len(d) for d in tokenized_docs], dtype=np.float64)
        self.avg_len = float(self.doc_len.mean()) if self.n else 0.0
        self.tf = [Counter(d) for d in tokenized_docs]

        # Document frequency per term, then smoothed idf.
        df = Counter()
        for d in tokenized_docs:
            for term in set(d):
                df[term] += 1
        self.idf = {
            term: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.n, dtype=np.float64)
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in range(self.n):
                freq = self.tf[i].get(term, 0)
                if freq == 0:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_len)
                scores[i] += idf * (freq * (self.k1 + 1)) / denom
        return scores
