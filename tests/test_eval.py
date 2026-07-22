"""Tests for retrieval evaluation metrics."""
import math

from ownai.eval import mrr, recall_at_k, evaluate_retrieval
from ownai.data import Chunk
from ownai.retrieval import HybridRetriever


def test_recall_at_k():
    # ranked lists of chunk ids; relevant id given per query
    assert recall_at_k([["a", "b", "c"]], [{"b"}], k=2) == 1.0
    assert recall_at_k([["a", "b", "c"]], [{"c"}], k=2) == 0.0
    assert recall_at_k([["a", "b"], ["x", "y"]], [{"b"}, {"z"}], k=2) == 0.5


def test_mrr():
    # relevant at rank 1 -> 1.0 ; rank 2 -> 0.5 ; absent -> 0
    assert mrr([["a", "b", "c"]], [{"a"}]) == 1.0
    assert mrr([["a", "b", "c"]], [{"b"}]) == 0.5
    assert math.isclose(mrr([["a", "b"], ["x", "y"]], [{"b"}, {"x"}]), (0.5 + 1.0) / 2)


def test_evaluate_retrieval_end_to_end():
    docs = [
        Chunk(id="star", title="Star", text="The Star makes Mario invincible."),
        Chunk(id="goomba", title="Goomba", text="A Goomba is a weak walking enemy."),
        Chunk(id="fire", title="Fire", text="The Fire Flower throws fireballs."),
    ]
    retr = HybridRetriever(embed_dim=16, seed=0)
    retr.index(docs, w2v_epochs=20)
    qa = [
        {"question": "how does mario become invincible", "relevant": ["star"]},
        {"question": "what throws fireballs", "relevant": ["fire"]},
    ]
    report = evaluate_retrieval(retr, qa, ks=(1, 3))
    assert 0.0 <= report["recall@1"] <= 1.0
    assert "recall@3" in report and "mrr" in report
    assert report["recall@3"] == 1.0  # both answers appear in top-3 of 3 docs
