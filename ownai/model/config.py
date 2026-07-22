"""Hyperparameter container for the mini-GPT."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128       # max context length (tokens)
    n_layer: int = 4            # number of transformer blocks
    n_head: int = 4             # attention heads per block
    n_embd: int = 128           # embedding / hidden width
    dropout: float = 0.1

    def __post_init__(self):
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

    def to_dict(self) -> dict:
        return asdict(self)
