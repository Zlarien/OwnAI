"""Byte-level Byte-Pair Encoding tokenizer, implemented from scratch.

The algorithm (same family as GPT-2's tokenizer):
  1. Start from the 256 raw byte values as the base vocabulary — this makes
     encoding lossless for ANY input, in any language, with no <unk> token.
  2. Repeatedly find the most frequent adjacent pair of tokens in the corpus
     and merge it into a new token, until the target vocab size is reached.
  3. To encode new text, apply the learned merges in the order they were found.

Special tokens (<|bos|>, <|eos|>, ...) get fixed ids just above the 256 byte
range and are never split.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

BYTE_VOCAB = 256


class BPETokenizer:
    def __init__(self, special_tokens: list[str] | None = None):
        self.special_tokens = list(special_tokens or [])
        # Special ids are assigned right after the byte range, before merges.
        self.special_to_id = {tok: BYTE_VOCAB + i for i, tok in enumerate(self.special_tokens)}
        self.id_to_special = {i: tok for tok, i in self.special_to_id.items()}
        self._merge_base = BYTE_VOCAB + len(self.special_tokens)
        # merges maps an ordered pair of ids -> the new merged id.
        self.merges: dict[tuple[int, int], int] = {}
        self._special_re = self._build_special_re()

    def _build_special_re(self):
        if not self.special_tokens:
            return None
        return re.compile("(" + "|".join(re.escape(t) for t in self.special_tokens) + ")")

    # ------------------------------------------------------------- properties
    @property
    def vocab_size(self) -> int:
        return self._merge_base + len(self.merges)

    def token_to_id(self, token: str) -> int:
        return self.special_to_id[token]

    # ---------------------------------------------------------------- training
    def train(self, corpus: list[str], vocab_size: int, verbose: bool = False) -> None:
        target_merges = vocab_size - self._merge_base
        if target_merges < 0:
            raise ValueError("vocab_size smaller than base (bytes + special tokens)")

        # Each document becomes a list of byte-ids; we merge across the corpus.
        sequences = [list(doc.encode("utf-8")) for doc in corpus]
        next_id = self._merge_base
        for _ in range(target_merges):
            pairs = Counter()
            for seq in sequences:
                for a, b in zip(seq, seq[1:]):
                    pairs[(a, b)] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            if pairs[best] < 2:
                break  # nothing worth merging anymore
            self.merges[best] = next_id
            sequences = [self._apply_merge(seq, best, next_id) for seq in sequences]
            next_id += 1
            if verbose and next_id % 500 == 0:
                print(f"  merges: {len(self.merges)}  vocab: {self.vocab_size}")

    @staticmethod
    def _apply_merge(seq: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        a, b = pair
        out, i, n = [], 0, len(seq)
        while i < n:
            if i < n - 1 and seq[i] == a and seq[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return out

    # ------------------------------------------------------------------ encode
    def _encode_bytes(self, data: bytes) -> list[int]:
        seq = list(data)
        # Apply merges greedily in the order they were learned.
        for pair, new_id in self.merges.items():
            seq = self._apply_merge(seq, pair, new_id)
        return seq

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.special_to_id["<|bos|>"])
        parts = self._special_re.split(text) if self._special_re else [text]
        for part in parts:
            if not part:
                continue
            if part in self.special_to_id:
                ids.append(self.special_to_id[part])
            else:
                ids.extend(self._encode_bytes(part.encode("utf-8")))
        if add_eos:
            ids.append(self.special_to_id["<|eos|>"])
        return ids

    # ------------------------------------------------------------------ decode
    def decode(self, ids: list[int]) -> str:
        # Reverse map: merged id -> its two component ids.
        inverse = {new_id: pair for pair, new_id in self.merges.items()}
        out = bytearray()
        specials = []  # (position_in_out, token_string) to splice back in
        for tid in ids:
            if tid in self.id_to_special:
                specials.append((len(out), self.id_to_special[tid]))
                continue
            stack = [tid]
            expanded = []
            while stack:
                cur = stack.pop()
                if cur in inverse:
                    a, b = inverse[cur]
                    stack.append(b)
                    stack.append(a)
                else:
                    expanded.append(cur)
            out.extend(expanded)
        # errors="replace": lossless for any validly-encoded text (no invalid
        # bytes occur), but tolerant of the arbitrary byte streams a generative
        # model can emit — decode must never crash.
        text = out.decode("utf-8", errors="replace")
        if not specials:
            return text
        # Re-insert special tokens. Byte offsets equal char offsets only for
        # ASCII, so rebuild by walking segments between special markers.
        result, prev = [], 0
        raw = bytes(out)
        for pos, tok in specials:
            result.append(raw[prev:pos].decode("utf-8", errors="replace"))
            result.append(tok)
            prev = pos
        result.append(raw[prev:].decode("utf-8", errors="replace"))
        return "".join(result)

    # --------------------------------------------------------------- persistence
    def save(self, path) -> None:
        path = Path(path)
        blob = {
            "special_tokens": self.special_tokens,
            # JSON keys must be strings: encode the pair as "a,b".
            "merges": [[f"{a},{b}", new_id] for (a, b), new_id in self.merges.items()],
        }
        path.write_text(json.dumps(blob, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "BPETokenizer":
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(special_tokens=blob["special_tokens"])
        merges = {}
        for key, new_id in blob["merges"]:
            a, b = key.split(",")
            merges[(int(a), int(b))] = new_id
        tok.merges = merges
        return tok
