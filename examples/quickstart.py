"""Quickstart: the whole OwnAI retrieval+answer flow in one runnable script.

This is the shortest path from raw domain files to grounded, cited answers,
using ONLY public `ownai` APIs — no CLI, no YAML, no network, no pretrained
anything. It exists so a newcomer (or a CI job) can clone the repo and, in a
few seconds, watch the from-scratch stack actually answer questions.

What it does, end to end:
    data/mario-wii/knowledge/*.md
        -> build_corpus_from_paths   (chunk the markdown into a JSONL corpus)
        -> build_index_from_corpus   (BM25 + hand-written word2vec = HybridRetriever)
        -> ExtractiveAnswerer        (pick the most relevant corpus sentences)

Extractive answering is used on purpose: every word comes verbatim from the
corpus, so the demo can never hallucinate and needs no trained mini-GPT.

Run it directly:
    python examples/quickstart.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from ownai.pipeline import build_corpus_from_paths, build_index_from_corpus
from ownai.rag import ExtractiveAnswerer

# The shipped offline knowledge base. Relative to the repo root so the script
# works regardless of the current working directory it is launched from.
_REPO_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = _REPO_ROOT / "data" / "mario-wii" / "knowledge"

# Three questions whose answers live in the corpus (gameplay + power-ups).
QUESTIONS = [
    "What does the Propeller Mushroom do?",
    "How many players can play at the same time?",
    "What does the Fire Flower do?",
]


def main() -> None:
    """Build the corpus + index, answer the example questions, print results."""
    # A throwaway corpus file: the index lives in memory, we only need the
    # JSONL on disk long enough to build the retriever from it.
    with tempfile.TemporaryDirectory() as tmp:
        corpus_path = Path(tmp) / "corpus.jsonl"

        # 1. Ingest the local markdown knowledge base into a chunked corpus.
        chunks = build_corpus_from_paths([KNOWLEDGE_DIR], corpus_path, max_words=160, overlap=30)
        print(f"Ingested {len(chunks)} chunks from {KNOWLEDGE_DIR}")

        # 2. Build the hybrid retriever. Few word2vec epochs keeps the demo fast;
        #    on this tiny corpus lexical BM25 already does most of the work.
        retriever = build_index_from_corpus(corpus_path, embed_dim=32, w2v_epochs=8, seed=0)
        print("Built hybrid retriever (BM25 + from-scratch word2vec)\n")

        # 3. Answer each question extractively and show the cited sources.
        answerer = ExtractiveAnswerer(retriever)
        for question in QUESTIONS:
            answer = answerer.answer(question, top_k=4, max_sentences=2)
            print(f"Q: {question}")
            print(f"A: {answer.text}")
            if answer.sources:
                print("   sources: " + "; ".join(answer.sources))
            print()


if __name__ == "__main__":
    main()
