"""Tests for retrieval: tokenization, BM25, word2vec embeddings, hybrid search."""
import numpy as np

from ownai.data import Chunk
from ownai.retrieval import BM25, Word2Vec, HybridRetriever, tokenize

DOCS = [
    Chunk(id="0", title="Goomba", text="The goomba is a weak enemy that walks slowly in Mario levels."),
    Chunk(id="1", title="Koopa", text="A koopa troopa is an enemy with a shell in Mario games."),
    Chunk(id="2", title="Star", text="The star power up makes Mario invincible for a short time."),
    Chunk(id="3", title="Jump", text="Mario can jump on enemies to defeat them in every level."),
]


def test_tokenize_lowercases_and_splits():
    assert tokenize("Mario's Star-Power!") == ["mario", "s", "star", "power"]


def test_bm25_ranks_relevant_doc_first():
    bm = BM25([tokenize(c.text) for c in DOCS])
    scores = bm.scores(tokenize("invincible star power"))
    assert int(np.argmax(scores)) == 2


def test_bm25_rare_term_scores_higher_than_common():
    bm = BM25([tokenize(c.text) for c in DOCS])
    # "mario" appears in every doc (idf ~0); "koopa" is rare (high idf).
    s_common = bm.scores(tokenize("mario"))
    s_rare = bm.scores(tokenize("koopa"))
    assert s_rare.max() > s_common.max()


def test_word2vec_learns_similarity():
    # A tiny corpus where two words always share context should end up close.
    sentences = [tokenize("mario jumps high"), tokenize("luigi jumps high")] * 200
    w2v = Word2Vec(dim=16, window=2, negative=5, seed=0)
    w2v.train(sentences, epochs=30, lr=0.05)
    # mario and luigi share identical contexts -> more similar than random pair.
    sim_related = w2v.similarity("mario", "luigi")
    sim_unrelated = w2v.similarity("mario", "high")
    assert sim_related > sim_unrelated


def test_word2vec_embed_sentence_shape():
    sentences = [tokenize(c.text) for c in DOCS]
    w2v = Word2Vec(dim=8, seed=0)
    w2v.train(sentences, epochs=2, lr=0.05)
    vec = w2v.embed_tokens(tokenize("mario star"))
    assert vec.shape == (8,)


def test_hybrid_retriever_returns_ranked_chunks():
    retr = HybridRetriever(embed_dim=16, seed=0)
    retr.index(DOCS, w2v_epochs=20)
    results = retr.search("how does mario become invincible", top_k=2)
    assert len(results) == 2
    ids = [c.id for c, _ in results]
    assert "2" in ids  # the star/invincible chunk must surface
    # scores are sorted descending
    assert results[0][1] >= results[1][1]


def test_hybrid_retriever_save_load(tmp_path):
    retr = HybridRetriever(embed_dim=8, seed=0)
    retr.index(DOCS, w2v_epochs=5)
    retr.save(tmp_path / "index")
    loaded = HybridRetriever.load(tmp_path / "index")
    q = "mario enemy shell"
    assert [c.id for c, _ in loaded.search(q, top_k=3)] == [
        c.id for c, _ in retr.search(q, top_k=3)
    ]
