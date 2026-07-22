"""Tests for answer-level evaluation (relevance + honest abstention)."""
from ownai.data import Chunk
from ownai.eval import evaluate_answers
from ownai.rag import ExtractiveAnswerer
from ownai.retrieval import HybridRetriever

DOCS = [
    Chunk(id="0", title="Star", text="The Star is a power-up. The Star makes Mario invincible for a short time. Enemies cannot hurt him.", source="wiki://Star"),
    Chunk(id="1", title="Goomba", text="A Goomba is a weak enemy. Mario defeats a Goomba by jumping on it.", source="wiki://Goomba"),
    Chunk(id="2", title="Fire Flower", text="The Fire Flower lets Mario throw fireballs at enemies.", source="wiki://Fire_Flower"),
]


def _answerer():
    retr = HybridRetriever(embed_dim=16, seed=0)
    retr.index(DOCS, w2v_epochs=20)
    return ExtractiveAnswerer(retr)


def test_evaluate_answers_perfect_scores():
    qa_pairs = [
        {"question": "How does Mario become invincible?", "expected_keywords": ["invincible"]},
        {"question": "What is a Goomba?", "expected_keywords": ["enemy", "goomba"]},
        {"question": "quantum chromodynamics tax law", "in_domain": False},
    ]
    report = evaluate_answers(_answerer(), qa_pairs)
    assert report["keyword_hit_rate"] == 1.0
    assert report["abstention_accuracy"] == 1.0
    assert report["n_in_domain"] == 2
    assert report["n_out_domain"] == 1


def test_evaluate_answers_empty_groups_are_none():
    # No in-domain-with-keywords pairs and no out-of-domain pairs -> both
    # metrics are undefined and reported as None, not a misleading number.
    report = evaluate_answers(_answerer(), [{"question": "What is a Goomba?"}])
    assert report["keyword_hit_rate"] is None
    assert report["abstention_accuracy"] is None
    assert report["n_in_domain"] == 1
    assert report["n_out_domain"] == 0
