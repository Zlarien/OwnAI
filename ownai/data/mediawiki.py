"""Download articles from any public MediaWiki site via its export API.

This is plain data acquisition (the same public API that powers "Export pages"
on wikis) — NOT an AI API. It returns raw wikitext Documents that clean.py then
turns into plain text. Kept import-light so the rest of the library doesn't
require network access.
"""
from __future__ import annotations

import time

from ownai.data.clean import clean_wikitext
from ownai.data.corpus import Document


def _get(session, api_url, params):
    params = {**params, "format": "json"}
    resp = session.get(api_url, params=params, timeout=30, headers={"User-Agent": "OwnAI/0.1 (educational project)"})
    resp.raise_for_status()
    return resp.json()


def list_category_pages(api_url: str, category: str, limit: int = 500) -> list[str]:
    """Return page titles belonging to a category (e.g. 'Category:Enemies')."""
    import requests

    session = requests.Session()
    titles, cont = [], {}
    while len(titles) < limit:
        data = _get(
            session,
            api_url,
            {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": category,
                "cmlimit": min(500, limit - len(titles)),
                **cont,
            },
        )
        titles.extend(m["title"] for m in data["query"]["categorymembers"])
        if "continue" in data:
            cont = data["continue"]
            time.sleep(0.2)
        else:
            break
    return titles


def fetch_pages(api_url: str, titles: list[str], clean: bool = True) -> list[Document]:
    """Fetch the wikitext of each page title and wrap it as a Document."""
    import requests

    session = requests.Session()
    docs = []
    for i in range(0, len(titles), 20):
        batch = titles[i : i + 20]
        data = _get(
            session,
            api_url,
            {
                "action": "query",
                "prop": "revisions",
                "rvprop": "content",
                "rvslots": "main",
                "titles": "|".join(batch),
            },
        )
        for page in data["query"]["pages"].values():
            if "revisions" not in page:
                continue
            raw = page["revisions"][0]["slots"]["main"]["*"]
            text = clean_wikitext(raw) if clean else raw
            if text.strip():
                docs.append(Document(title=page["title"], text=text, source=f"{api_url}?title={page['title']}"))
        time.sleep(0.2)
    return docs
