"""End-to-end smoke test of the offline pipeline on a tiny local corpus.

Proves the whole chain wires together: local files -> chunks -> hybrid index
-> extractive answer with sources. No network, no pretrained anything.
"""
from ownai.pipeline import build_corpus_from_paths, build_index_from_corpus
from ownai.rag import ExtractiveAnswerer


def test_offline_pipeline_answers_with_sources(tmp_path):
    # 1. Author a tiny knowledge base.
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "star.md").write_text(
        "# Star\nThe Star is a power-up. It makes Mario invincible for a short time.",
        encoding="utf-8",
    )
    (kb / "goomba.md").write_text(
        "# Goomba\nA Goomba is a weak enemy. Mario defeats a Goomba by jumping on it.",
        encoding="utf-8",
    )

    # 2. Ingest -> corpus on disk.
    corpus_path = tmp_path / "corpus.jsonl"
    chunks = build_corpus_from_paths([kb], corpus_path, max_words=60, overlap=10)
    assert corpus_path.exists()
    assert len(chunks) >= 2

    # 3. Build the retrieval index.
    retr = build_index_from_corpus(corpus_path, embed_dim=16, w2v_epochs=15, seed=0)

    # 4. Ask a question -> grounded answer + citation.
    ans = ExtractiveAnswerer(retr).answer("how does mario become invincible", max_sentences=1)
    assert "invincible" in ans.text.lower()
    assert ans.sources
