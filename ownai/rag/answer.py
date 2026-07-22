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


NOT_FOUND = (
    "I couldn't find anything about that in my knowledge base. Try rephrasing "
    "with more specific keywords — and note that I only know this one domain, "
    "and I answer in the language of the corpus."
)


@dataclass
class Answer:
    text: str
    sources: list[str] = field(default_factory=list)
    contexts: list[Chunk] = field(default_factory=list)


# Compact bilingual (EN + FR) function-word list. These carry no topic meaning
# even when rare in the corpus (e.g. a "how"/"comment" that appears only once).
# Our word tokenizer also splits French elisions (d', l', qu') into d/l/qu.
STOPWORDS = frozenset(
    """
    a an the is are was were be been being do does did done has have had how what who whom
    whose when where why which that this these those of in on at to from with without and or
    not it its as for by can could will would should about into your you i we they he she
    make makes made get gets give gives use used
    le la les un une des de du est sont etre être ete été comment qui que quoi quel quelle quels
    quelles quand ou où pourquoi dans sur a au aux et ne pas ce cet cette ces son sa ses pour par
    avec plus tres très d l j c qu s n m t on se leur
    """.split()
)


def _informative_terms(q_tokens: set[str], idf: dict | None) -> set[str]:
    """Query terms that actually carry topic meaning in THIS corpus.

    A term is informative if it is (a) not a function word, (b) present in the
    corpus, and (c) not among the most common corpus words (low IDF, like
    "mario" or "the"). If the query shares none of these with any answer
    sentence, we have not really found an answer and should say so.
    """
    if not idf:
        return set()
    values = sorted(idf.values())
    # Exclude only the most common ~20% of vocabulary (the low-IDF tail).
    low_bar = values[len(values) // 5]
    return {
        t
        for t in q_tokens
        if t not in STOPWORDS and idf.get(t, 0.0) >= low_bar and t in idf
    }


def _overlap_score(q_tokens: set[str], sentence: str, idf: dict | None = None) -> float:
    """Query/sentence overlap, IDF-weighted so rare words dominate stopwords.

    Without idf this is plain shared-word count / sqrt(length). With idf (the
    BM25 inverse document frequencies), matching a rare term like "comet"
    counts far more than matching "is" or "the" — which is what makes the
    selected sentence actually answer the question.
    """
    s_tokens = set(tokenize(sentence))
    if not s_tokens:
        return 0.0
    shared = q_tokens & s_tokens
    if idf is None:
        weight = float(len(shared))
    else:
        weight = sum(idf.get(t, 0.0) for t in shared)
    return weight / (len(s_tokens) ** 0.5)


class ExtractiveAnswerer:
    """Answer by selecting the most query-relevant sentences from top chunks."""

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def answer(self, question: str, top_k: int = 4, max_sentences: int = 2) -> Answer:
        hits = self.retriever.search(question, top_k=top_k)
        if not hits:
            return Answer(text="I don't have information on that in my knowledge base.")
        q_tokens = set(tokenize(question))
        idf = getattr(self.retriever, "bm25", None)
        idf = idf.idf if idf is not None else None

        # Collect unique sentences. Overlapping chunks repeat sentences, and
        # chunk boundaries create fragments that are prefixes of a full
        # sentence — keep only the longest form of each.
        by_key = {}
        for chunk, _ in hits:
            for sent in split_sentences(chunk.text):
                key = sent.lower()
                if key not in by_key:
                    by_key[key] = (sent, chunk)

        kept = []
        keys = sorted(by_key, key=len, reverse=True)  # longest first
        for key in keys:
            if any(key != other and key in other for other in (k for k, _ in kept)):
                continue  # this sentence is a substring of a longer kept one
            kept.append((key, by_key[key]))

        scored = [
            (_overlap_score(q_tokens, sent, idf), sent, chunk) for _, (sent, chunk) in kept
        ]
        scored.sort(key=lambda t: -t[0])

        # Confidence gate: keep only sentences that share an informative (rare,
        # non-stopword) query term with the corpus. If none qualify, the query
        # is out-of-domain or in the wrong language — admit it, don't guess.
        informative = _informative_terms(q_tokens, idf)
        relevant = [
            (score, sent, chunk)
            for score, sent, chunk in scored
            if informative & set(tokenize(sent))
        ]
        if not relevant:
            return Answer(text=NOT_FOUND)
        chosen = relevant[:max_sentences]

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
