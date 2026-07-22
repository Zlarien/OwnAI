"""Answer-level evaluation: does the pipeline actually *answer* well?

Retrieval metrics (Recall@k, MRR) only measure whether the right chunk was
fetched. They say nothing about the final answer the user reads. This module
scores the answerer's output directly on two axes that matter for a domain AI:

- keyword_hit_rate: on in-domain questions, does the answer mention what it
  should? (a cheap, corpus-agnostic proxy for "is this answer relevant?")
- abstention_accuracy: on out-of-domain questions, does it honestly refuse
  instead of hallucinating? A domain AI that guesses confidently is worse than
  one that says "I don't know."

Together they turn "it feels right" into two defensible numbers.
"""
from __future__ import annotations

from ownai.rag.answer import NOT_FOUND


def _abstains(answer) -> bool:
    """True if the answerer declined to answer.

    Two honest signals of abstention: it returned the canonical NOT_FOUND
    message, or it cited no sources (an answer with no provenance is, by this
    pipeline's contract, a "not found").
    """
    return answer.text == NOT_FOUND or answer.sources == []


def evaluate_answers(answerer, qa_pairs) -> dict:
    """Score an answerer on relevance (in-domain) and honesty (out-of-domain).

    qa_pairs: list of dicts with keys
        question:         str
        expected_keywords: [str]  (optional; only used for in-domain pairs)
        in_domain:        bool    (default True)

    Returns a dict with:
        keyword_hit_rate:     over in-domain pairs that carry expected_keywords,
                              the fraction whose (lowercased) answer text
                              contains at least one expected keyword.
        abstention_accuracy:  over out-of-domain pairs, the fraction where the
                              answerer correctly abstained.
        n_in_domain:          count of in-domain pairs.
        n_out_domain:         count of out-of-domain pairs.

    Empty-group choice: a rate over zero eligible pairs is undefined, so we
    report ``None`` rather than a misleading 0.0 (which reads as "0% correct")
    or 1.0 (which reads as "perfect"). None makes "not measured" explicit.
    """
    n_in_domain = n_out_domain = 0
    kw_eligible = kw_hits = 0
    abst_total = abst_correct = 0

    for pair in qa_pairs:
        in_domain = pair.get("in_domain", True)
        if in_domain:
            n_in_domain += 1
            keywords = pair.get("expected_keywords")
            if keywords:
                kw_eligible += 1
                text = answerer.answer(pair["question"]).text.lower()
                if any(kw.lower() in text for kw in keywords):
                    kw_hits += 1
        else:
            n_out_domain += 1
            abst_total += 1
            if _abstains(answerer.answer(pair["question"])):
                abst_correct += 1

    return {
        "keyword_hit_rate": (kw_hits / kw_eligible) if kw_eligible else None,
        "abstention_accuracy": (abst_correct / abst_total) if abst_total else None,
        "n_in_domain": n_in_domain,
        "n_out_domain": n_out_domain,
    }
