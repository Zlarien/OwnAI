"""OwnGPT's own byte-level BPE tokenizer: trained here, encoded fast by tiktoken.

Training is ours (Python, below): split text into word-like chunks with a regex,
start from the 256 raw bytes, and repeatedly merge the most frequent adjacent
pair until the vocabulary is full. The result is a list of merges.

Encoding billions of tokens in Python would take days, so the learned merges are
handed to tiktoken's Rust engine as ``mergeable_ranks``. Byte-level BPE applied
by merge rank is the same algorithm GPT-2/GPT-4 use, so tiktoken encodes with
*our* vocabulary, it just does it fast.

Special tokens (never produced from raw text):
  <|endoftext|>  separates documents, ends every assistant reply
  <|user|>       starts a user turn
  <|assistant|>  starts an assistant turn
"""
from __future__ import annotations

import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# GPT-4 style split without the English-only contractions: letters stay with an
# optional leading punctuation/space, numbers go in groups of up to 3 digits.
PATTERN = r"""[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
SPECIALS = ["<|endoftext|>", "<|user|>", "<|assistant|>"]


def _merge_word(word, pair, new_id):
    out, i = [], 0
    a, b = pair
    while i < len(word):
        if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


def train_bpe(texts, vocab_size: int = 32768, max_chars: int | None = None, lang: str = "fr",
              pattern: str = PATTERN, verbose: bool = True) -> "Tokenizer":
    """Learn ``vocab_size - 256 - len(SPECIALS)`` merges from an iterable of texts."""
    import regex

    split = regex.compile(pattern)
    counts: Counter = Counter()
    seen = 0
    for text in texts:
        counts.update(split.findall(text))
        seen += len(text)
        if max_chars and seen >= max_chars:
            break
    if verbose:
        print(f"tokenizer: {seen:,} chars, {len(counts):,} unique chunks", flush=True)

    words = [list(chunk.encode("utf-8")) for chunk in counts]
    freqs = list(counts.values())
    del counts
    pair_counts: dict = defaultdict(int)
    where: dict = defaultdict(set)  # pair -> indices of words that (may) contain it
    for i, (w, f) in enumerate(zip(words, freqs)):
        for pair in zip(w, w[1:]):
            pair_counts[pair] += f
            where[pair].add(i)
    heap = [(-c, pair) for pair, c in pair_counts.items()]
    heapq.heapify(heap)

    vocab = [bytes([i]) for i in range(256)]
    known = {b: i for i, b in enumerate(vocab)}
    merges = []  # (a, b) -> id 256 + index, in training order
    target = vocab_size - len(SPECIALS)
    while len(vocab) < target and heap:
        neg, pair = heapq.heappop(heap)
        count = pair_counts.get(pair, 0)
        if count <= 0:
            continue
        if -neg != count:  # stale heap entry: push the current count back
            heapq.heappush(heap, (-count, pair))
            continue
        new_bytes = vocab[pair[0]] + vocab[pair[1]]
        if new_bytes in known:  # same bytes reachable by another merge path: reuse the id
            new_id = known[new_bytes]
        else:
            new_id = len(vocab)
            vocab.append(new_bytes)
            known[new_bytes] = new_id
            merges.append(pair)
        touched = set()
        for i in where.pop(pair, ()):
            w, f = words[i], freqs[i]
            merged = _merge_word(w, pair, new_id)
            if len(merged) == len(w):
                continue
            for q in zip(w, w[1:]):
                pair_counts[q] -= f
            for q in zip(merged, merged[1:]):
                pair_counts[q] += f
                where[q].add(i)
                touched.add(q)
            words[i] = merged
        pair_counts.pop(pair, None)
        for q in touched:
            if pair_counts.get(q, 0) > 0:
                heapq.heappush(heap, (-pair_counts[q], q))
        if verbose and len(vocab) % 2000 == 0:
            print(f"  {len(vocab):,} tokens, last merge {new_bytes!r} x{count:,}", flush=True)
    return Tokenizer({"pattern": pattern, "merges": [list(m) for m in merges], "specials": SPECIALS,
                      "lang": lang})


class Tokenizer:
    """Our vocabulary, encoded by tiktoken. Serializes to a small JSON dict."""

    def __init__(self, spec: dict):
        import tiktoken

        self.spec = spec
        vocab = [bytes([i]) for i in range(256)]
        for a, b in spec["merges"]:
            vocab.append(vocab[a] + vocab[b])
        self.vocab = vocab
        self.special_ids = {s: len(vocab) + i for i, s in enumerate(spec["specials"])}
        self.enc = tiktoken.Encoding(
            name="owngpt",
            pat_str=spec["pattern"],
            mergeable_ranks={b: i for i, b in enumerate(vocab)},
            special_tokens=self.special_ids,
        )
        self.eot = self.special_ids["<|endoftext|>"]
        self.user = self.special_ids["<|user|>"]
        self.assistant = self.special_ids["<|assistant|>"]
        self.n_vocab = len(vocab) + len(self.special_ids)
        self.lang = spec.get("lang", "fr")

    def encode(self, text: str) -> list[int]:
        """Raw text only: special-token strings typed by a user stay plain text."""
        return self.enc.encode_ordinary(text)

    def encode_batch(self, texts) -> list[list[int]]:
        return self.enc.encode_ordinary_batch(list(texts))

    def decode(self, ids) -> str:
        return self.enc.decode([i for i in ids if i < len(self.vocab)])

    def token_bytes(self) -> np.ndarray:
        """UTF-8 byte length of every token id (0 for specials), for bits-per-byte."""
        return np.array([len(b) for b in self.vocab] + [0] * len(self.special_ids), dtype=np.int64)

    def to_dict(self) -> dict:
        return self.spec

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.spec), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Tokenizer":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def __eq__(self, other):
        return isinstance(other, Tokenizer) and self.spec["merges"] == other.spec["merges"]
