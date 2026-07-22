"""Retrieval quality metrics: Recall@k and Mean Reciprocal Rank.

These turn "the chatbot feels okay" into defensible numbers — the difference
between a toy and a portfolio project a recruiter can trust.
"""
from __future__ import annotations


def recall_at_k(ranked_lists, relevant_sets, k: int) -> float:
    """Fraction of queries whose top-k contains at least one relevant id."""
    if not ranked_lists:
        return 0.0
    hits = 0
    for ranked, relevant in zip(ranked_lists, relevant_sets):
        if set(ranked[:k]) & set(relevant):
            hits += 1
    return hits / len(ranked_lists)


def mrr(ranked_lists, relevant_sets) -> float:
    """Mean reciprocal rank of the first relevant id across queries."""
    if not ranked_lists:
        return 0.0
    total = 0.0
    for ranked, relevant in zip(ranked_lists, relevant_sets):
        relevant = set(relevant)
        for rank, item in enumerate(ranked, start=1):
            if item in relevant:
                total += 1.0 / rank
                break
    return total / len(ranked_lists)


def evaluate_retrieval(retriever, qa_pairs, ks=(1, 3, 5), top_k=None) -> dict:
    """Run a retriever over a QA set and compute Recall@k for each k, plus MRR.

    qa_pairs: list of {"question": str, "relevant": [chunk_id, ...]}.
    """
    top_k = top_k or max(ks)
    ranked_lists, relevant_sets = [], []
    for pair in qa_pairs:
        hits = retriever.search(pair["question"], top_k=top_k)
        ranked_lists.append([c.id for c, _ in hits])
        relevant_sets.append(set(pair["relevant"]))

    report = {f"recall@{k}": recall_at_k(ranked_lists, relevant_sets, k) for k in ks}
    report["mrr"] = mrr(ranked_lists, relevant_sets)
    report["n_queries"] = len(qa_pairs)
    return report
