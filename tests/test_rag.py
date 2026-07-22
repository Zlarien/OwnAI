"""Tests for the RAG answering pipeline."""
from ownai.data import Chunk
from ownai.rag import ExtractiveAnswerer, build_prompt, split_sentences
from ownai.retrieval import HybridRetriever

DOCS = [
    Chunk(id="0", title="Star", text="The Star is a power-up. The Star makes Mario invincible for a short time. Enemies cannot hurt him.", source="wiki://Star"),
    Chunk(id="1", title="Goomba", text="A Goomba is a weak enemy. Mario defeats a Goomba by jumping on it.", source="wiki://Goomba"),
    Chunk(id="2", title="Fire Flower", text="The Fire Flower lets Mario throw fireballs at enemies.", source="wiki://Fire_Flower"),
]


def test_split_sentences():
    s = split_sentences("The Star is great. Mario wins! Does it work? Yes.")
    assert s == ["The Star is great.", "Mario wins!", "Does it work?", "Yes."]


def test_build_prompt_includes_context_and_question():
    prompt = build_prompt("How does Mario become invincible?", [DOCS[0]])
    assert "invincible" in prompt.lower()
    assert "How does Mario become invincible?" in prompt
    assert "Star" in prompt


def test_extractive_answerer_selects_relevant_sentence():
    retr = HybridRetriever(embed_dim=16, seed=0)
    retr.index(DOCS, w2v_epochs=20)
    ans = ExtractiveAnswerer(retr)
    result = ans.answer("How does Mario become invincible?", top_k=2, max_sentences=1)
    assert "invincible" in result.text.lower()
    assert result.sources  # provenance returned
    assert any("Star" in s for s in result.sources)


def test_extractive_answer_has_no_duplicate_sentences():
    # Overlapping chunks can surface the same sentence twice; the answer must
    # not repeat it, and sources must be unique.
    overlapping = [
        Chunk(id="a", title="Koopalings", text="The Koopalings are seven. Larry is one of them.", source="s"),
        Chunk(id="b", title="Koopalings", text="Larry is one of them. They serve Bowser.", source="s"),
    ]
    retr = HybridRetriever(embed_dim=16, seed=0)
    retr.index(overlapping, w2v_epochs=10)
    ans = ExtractiveAnswerer(retr).answer("who is Larry", top_k=2, max_sentences=3)
    sentences = split_sentences(ans.text)
    assert len(sentences) == len(set(sentences))
    assert len(ans.sources) == len(set(ans.sources))


def test_extractive_answerer_handles_no_match_gracefully():
    retr = HybridRetriever(embed_dim=16, seed=0)
    retr.index(DOCS, w2v_epochs=10)
    ans = ExtractiveAnswerer(retr)
    result = ans.answer("completely unrelated quantum chromodynamics", top_k=2)
    # Still returns a structured result, never crashes.
    assert isinstance(result.text, str)
    assert result.text  # non-empty (best-effort passage)
