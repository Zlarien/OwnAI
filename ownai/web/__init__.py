"""Standard-library-only web chat UI for the domain AI.

No Flask/FastAPI/any external dependency — just http.server. The whole point of
OwnAI is a from-scratch stack, so even the web layer leans on nothing beyond the
Python standard library.
"""
from ownai.web.server import PAGE_HTML, answer_to_json, build_handler, serve

__all__ = [
    "PAGE_HTML",
    "answer_to_json",
    "build_handler",
    "serve",
]
