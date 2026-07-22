"""Word tokenization for retrieval (distinct from the model's BPE tokenizer).

Lowercases and keeps alphanumeric runs. This is what BM25 and word2vec operate
on — a bag of normalized words, language-agnostic enough for EN/FR corpora.
"""
from __future__ import annotations

import re

_WORD = re.compile(r"[0-9a-zà-öø-ÿ]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())
