"""Tests for corpus building: cleaning, chunking, JSONL persistence.

Network fetching (MediaWiki) is not tested here — it is isolated behind a thin
function. We test the pure transformation logic that turns raw documents into
retrieval-ready chunks.
"""
import json

from ownai.data import (
    Chunk,
    chunk_document,
    clean_wikitext,
    read_corpus,
    write_corpus,
)


def test_clean_wikitext_strips_markup():
    raw = "'''Mario''' is a [[character]] in {{game}} the <ref>note</ref> series."
    out = clean_wikitext(raw)
    assert "'''" not in out
    assert "[[" not in out and "]]" not in out
    assert "{{" not in out
    assert "<ref>" not in out
    assert "Mario is a character in  the  series." in out or "Mario is a character" in out


def test_clean_wikitext_keeps_link_labels():
    assert "goomba" in clean_wikitext("A [[enemy|goomba]] appears.").lower()


def test_chunk_document_respects_size_and_overlap():
    words = " ".join(f"w{i}" for i in range(250))
    chunks = chunk_document(
        title="Doc", text=words, source="url://doc", max_words=100, overlap=20
    )
    assert len(chunks) >= 3
    for c in chunks:
        assert isinstance(c, Chunk)
        assert len(c.text.split()) <= 100
        assert c.title == "Doc"
        assert c.source == "url://doc"
    # Overlap: consecutive chunks share tail/head words.
    first_tail = chunks[0].text.split()[-20:]
    second_head = chunks[1].text.split()[:20]
    assert first_tail == second_head


def test_chunk_document_short_text_single_chunk():
    chunks = chunk_document(title="T", text="only a few words", source="s", max_words=100)
    assert len(chunks) == 1
    assert chunks[0].id == "T#0"


def test_write_and_read_corpus_roundtrip(tmp_path):
    chunks = [
        Chunk(id="A#0", title="A", text="first chunk", source="s1"),
        Chunk(id="B#0", title="B", text="second chunk", source="s2"),
    ]
    path = tmp_path / "corpus.jsonl"
    write_corpus(chunks, path)
    # Each line is a standalone JSON object.
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["id"] == "A#0"
    loaded = read_corpus(path)
    assert [c.id for c in loaded] == ["A#0", "B#0"]
    assert loaded[1].text == "second chunk"


def test_read_local_text_files(tmp_path):
    from ownai.data import documents_from_paths

    (tmp_path / "a.txt").write_text("Alpha content here.", encoding="utf-8")
    (tmp_path / "b.md").write_text("# Beta\n\nBeta content.", encoding="utf-8")
    docs = documents_from_paths([tmp_path])
    titles = sorted(d.title for d in docs)
    assert titles == ["a", "b"]
    assert any("Beta content" in d.text for d in docs)
