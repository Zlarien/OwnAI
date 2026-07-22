"""Tests for the standard-library web chat UI.

These never bind a socket: they build a tiny in-memory retriever, wrap it in an
answerer, and exercise the pure functions (answer_to_json, PAGE_HTML, the
handler factory) directly. No real server is started.
"""
from http.server import BaseHTTPRequestHandler

from ownai.data import Chunk
from ownai.rag import ExtractiveAnswerer
from ownai.retrieval import HybridRetriever
from ownai.web import PAGE_HTML, answer_to_json, build_handler

# Same corpus pattern as tests/test_rag.py.
DOCS = [
    Chunk(id="0", title="Star", text="The Star is a power-up. The Star makes Mario invincible for a short time. Enemies cannot hurt him.", source="wiki://Star"),
    Chunk(id="1", title="Goomba", text="A Goomba is a weak enemy. Mario defeats a Goomba by jumping on it.", source="wiki://Goomba"),
    Chunk(id="2", title="Fire Flower", text="The Fire Flower lets Mario throw fireballs at enemies.", source="wiki://Fire_Flower"),
]


def _answerer():
    retr = HybridRetriever(embed_dim=16, seed=0)
    retr.index(DOCS, w2v_epochs=20)
    return ExtractiveAnswerer(retr)


def test_answer_to_json_keys_and_types():
    result = answer_to_json(_answerer(), "How does Mario become invincible?", top_k=2)
    assert set(result.keys()) == {"answer", "sources"}
    assert isinstance(result["answer"], str)
    assert isinstance(result["sources"], list)
    assert all(isinstance(s, str) for s in result["sources"])
    # A real in-domain question yields a non-empty, faithful answer.
    assert "invincible" in result["answer"].lower()


def test_answer_to_json_is_json_serializable():
    import json

    result = answer_to_json(_answerer(), "what is a Goomba", top_k=3)
    # Round-trips cleanly -> safe to send over the wire.
    assert json.loads(json.dumps(result)) == result


def test_page_html_is_self_contained_and_branded():
    assert "<input" in PAGE_HTML
    assert "OwnAI" in PAGE_HTML
    # Theme-aware and dependency-free.
    assert "prefers-color-scheme" in PAGE_HTML
    assert "http://" not in PAGE_HTML  # no external CDN/fonts
    assert "https://" not in PAGE_HTML


def test_build_handler_returns_handler_class():
    handler = build_handler(_answerer(), top_k=2)
    assert isinstance(handler, type)
    assert issubclass(handler, BaseHTTPRequestHandler)
