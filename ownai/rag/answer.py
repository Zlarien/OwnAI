"""Answering pipeline: retrieve context, then answer either extractively
(pick the best sentences from the corpus — always faithful) or generatively
(let the mini-GPT write, conditioned on the retrieved context).

The extractive mode is the reliable default: it can never hallucinate because
every word comes verbatim from the domain corpus. The generative mode shows off
the from-scratch transformer. Both cite their sources.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from ownai.data import Chunk
from ownai.retrieval import HybridRetriever
from ownai.retrieval.tokenize import tokenize

_SENT = re.compile(r"[^.!?]*[.!?]+|\S[^.!?]*$")


def split_sentences(text: str) -> list[str]:
    return [m.group().strip() for m in _SENT.finditer(text) if m.group().strip()]


def build_prompt(question: str, chunks: list[Chunk]) -> str:
    """Assemble a RAG prompt: retrieved context followed by the question."""
    context = "\n".join(f"- ({c.title}) {c.text}" for c in chunks)
    return (
        "Context:\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


@dataclass
class Answer:
    text: str
    sources: list[str] = field(default_factory=list)
    contexts: list[Chunk] = field(default_factory=list)


def _overlap_score(q_tokens: set[str], sentence: str) -> float:
    s_tokens = set(tokenize(sentence))
    if not s_tokens:
        return 0.0
    return len(q_tokens & s_tokens) / (len(s_tokens) ** 0.5)


class ExtractiveAnswerer:
    """Answer by selecting the most query-relevant sentences from top chunks."""

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def answer(self, question: str, top_k: int = 4, max_sentences: int = 2) -> Answer:
        hits = self.retriever.search(question, top_k=top_k)
        if not hits:
            return Answer(text="I don't have information on that in my knowledge base.")
        q_tokens = set(tokenize(question))

        scored, seen_sents = [], set()
        for chunk, _ in hits:
            for sent in split_sentences(chunk.text):
                key = sent.lower()
                if key in seen_sents:  # overlapping chunks repeat sentences
                    continue
                seen_sents.add(key)
                scored.append((_overlap_score(q_tokens, sent), sent, chunk))
        scored.sort(key=lambda t: -t[0])

        chosen = scored[:max_sentences] if scored else []
        # Fallback: if nothing overlaps, return the single best chunk's first sentence.
        if not chosen or chosen[0][0] == 0.0:
            top_chunk = hits[0][0]
            first = split_sentences(top_chunk.text)[:1] or [top_chunk.text]
            return Answer(
                text=" ".join(first),
                sources=[f"{top_chunk.title} ({top_chunk.source})"],
                contexts=[top_chunk],
            )

        text = " ".join(s for _, s, _ in chosen)
        seen_src, sources, contexts = set(), [], []
        for _, _, chunk in chosen:
            label = f"{chunk.title} ({chunk.source})"
            if label not in seen_src:
                seen_src.add(label)
                sources.append(label)
                contexts.append(chunk)
        return Answer(text=text, sources=sources, contexts=contexts)


class GenerativeAnswerer:
    """Answer with the from-scratch mini-GPT, conditioned on retrieved context.

    Needs a trained model + the BPE tokenizer used to train it. The output is
    genuinely generated token-by-token; it can be imperfect on a small corpus,
    which is why the sources are always shown alongside.
    """

    def __init__(self, retriever: HybridRetriever, model, tokenizer, max_new_tokens: int = 60):
        self.retriever = retriever
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def answer(self, question: str, top_k: int = 4, temperature: float = 0.8, seed: int = 0) -> Answer:
        hits = self.retriever.search(question, top_k=top_k)
        contexts = [c for c, _ in hits]
        prompt = build_prompt(question, contexts)
        ids = self.tokenizer.encode(prompt)[-self.model.cfg.block_size :]
        out = self.model.generate(
            np.array([ids]), max_new_tokens=self.max_new_tokens, temperature=temperature, seed=seed
        )
        generated = out[0, len(ids) :].tolist()
        text = self.tokenizer.decode(generated).split("\n")[0].strip()
        sources = [f"{c.title} ({c.source})" for c in contexts]
        return Answer(text=text or "(no answer generated)", sources=sources, contexts=contexts)
