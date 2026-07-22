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


def test_extractive_drops_sentence_fragments():
    # Chunk boundaries can create a truncated fragment that is a prefix of a
    # full sentence. The answer must not contain the redundant fragment.
    chunks = [
        Chunk(id="a", title="Neptune", text="Neptune and Uranus are bluish ice giants", source="s"),
        Chunk(id="b", title="Neptune", text="Neptune and Uranus are bluish ice giants. Uranus spins on its side.", source="s"),
        # Filler docs so "bluish"/"ice"/"giants" are informative (rare), not ubiquitous.
        Chunk(id="c", title="Star", text="The star makes Mario invincible.", source="s"),
        Chunk(id="d", title="Goomba", text="A goomba is a weak enemy that walks.", source="s"),
    ]
    retr = HybridRetriever(embed_dim=16, seed=0)
    retr.index(chunks, w2v_epochs=10)
    ans = ExtractiveAnswerer(retr).answer("what are the bluish ice giants", top_k=2, max_sentences=3)
    # The fragment (no period) must not appear twice-worth of "ice giants".
    assert ans.text.lower().count("bluish ice giants") == 1


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


def test_extractive_prefers_rare_term_over_stopwords():
    # The wrong sentence shares only common words with the query; the right one
    # shares the rare, meaningful term. IDF weighting must pick the right one.
    # A realistic corpus: "what/is/a/the" appear everywhere (low IDF), "comet"
    # is rare (high IDF). The short stopword-heavy sentence would win under
    # naive overlap; IDF weighting must pick the meaningful comet sentence.
    # "what/is/a" appear in EVERY document (low IDF, as real stopwords are);
    # only "comet" is rare. The short "what is a star" sentence would win under
    # naive overlap; IDF weighting must pick the meaningful comet sentence.
    docs = [
        Chunk(id="0", title="Planet", text="What is a planet here. A planet is a large world.", source="s"),
        Chunk(id="1", title="Star", text="What is a star here. A star is a hot ball of gas.", source="s"),
        Chunk(id="2", title="Moon", text="What is a moon here. A moon is a natural satellite.", source="s"),
        Chunk(id="3", title="Sun", text="What is a sun here. A sun is a bright star.", source="s"),
        # The answer sentence shares only {a, is, comet} — no leading "what".
        Chunk(id="4", title="Comet", text="A comet is a body of ice and dust travelling in deep space.", source="s"),
    ]
    retr = HybridRetriever(embed_dim=16, seed=0)
    retr.index(docs, w2v_epochs=10)
    ans = ExtractiveAnswerer(retr).answer("what is a comet", top_k=5, max_sentences=1)
    # Naive overlap would pick a short "What is a <x> here" stopword sentence;
    # IDF weighting picks the comet sentence because "comet" is the rare term.
    assert "comet" in ans.text.lower()


def test_extractive_says_dont_know_instead_of_guessing():
    # An out-of-domain query shares no meaningful term with the corpus. The
    # answerer must admit it, NOT confidently return a random corpus sentence.
    retr = HybridRetriever(embed_dim=16, seed=0)
    retr.index(DOCS, w2v_epochs=10)
    ans = ExtractiveAnswerer(retr).answer("quantum chromodynamics tax law", top_k=3)
    assert ans.sources == []  # no source cited -> honest "not found"
    for doc in DOCS:
        assert doc.text not in ans.text  # did not parrot a corpus sentence


def test_cross_lingual_query_is_refused_not_hallucinated():
    # French question against an English corpus: only stopwords/names could
    # match. Must refuse rather than return an unrelated English sentence.
    retr = HybridRetriever(embed_dim=16, seed=0)
    retr.index(DOCS, w2v_epochs=10)
    ans = ExtractiveAnswerer(retr).answer("Comment Mario tire des boules de feu", top_k=3)
    assert ans.sources == []


def test_informative_query_still_answers():
    # A query with a rare, in-corpus term ("invincible") must still answer.
    retr = HybridRetriever(embed_dim=16, seed=0)
    retr.index(DOCS, w2v_epochs=20)
    ans = ExtractiveAnswerer(retr).answer("what makes Mario invincible", top_k=3)
    assert "invincible" in ans.text.lower()
    assert ans.sources


def test_question_word_alone_does_not_make_it_confident():
    # "how" is rare in the corpus (high IDF) but is a function word; a match on
    # "how" alone must NOT count as finding an answer.
    docs = [
        Chunk(id="a", title="Guide", text="This shows you how to finish the level.", source="s"),
        Chunk(id="b", title="Goomba", text="A goomba is a weak enemy.", source="s"),
    ]
    retr = HybridRetriever(embed_dim=16, seed=0)
    retr.index(docs, w2v_epochs=10)
    ans = ExtractiveAnswerer(retr).answer("how do dragons breathe", top_k=2)
    assert ans.sources == []  # 'how' must not trigger a confident answer


def test_answer_drops_trailing_irrelevant_sentence():
    docs = [
        Chunk(id="a", title="Star", text="The Star makes Mario invincible.", source="s1"),
        Chunk(id="b", title="Guide", text="A green block lets Luigi finish the level for you.", source="s2"),
    ]
    retr = HybridRetriever(embed_dim=16, seed=0)
    retr.index(docs, w2v_epochs=10)
    ans = ExtractiveAnswerer(retr).answer("what makes Mario invincible", top_k=2, max_sentences=2)
    # Only the invincible sentence is relevant; the Luigi sentence must be dropped.
    assert "invincible" in ans.text.lower()
    assert "luigi" not in ans.text.lower()
    assert len(ans.sources) == 1
