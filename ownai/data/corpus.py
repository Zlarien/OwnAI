"""Corpus data model: Document -> Chunks -> JSONL on disk.

A Document is one source article. Chunks are the retrieval units: overlapping
word windows carrying their provenance (title + source), so the chatbot can
cite where an answer came from.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Document:
    title: str
    text: str
    source: str = ""


@dataclass
class Chunk:
    id: str
    title: str
    text: str
    source: str = ""


def chunk_document(
    title: str, text: str, source: str = "", max_words: int = 180, overlap: int = 30
) -> list[Chunk]:
    """Split text into overlapping windows of at most `max_words` words."""
    words = text.split()
    if not words:
        return []
    if len(words) <= max_words:
        return [Chunk(id=f"{title}#0", title=title, text=" ".join(words), source=source)]

    step = max(1, max_words - overlap)
    chunks, idx, start = [], 0, 0
    while start < len(words):
        window = words[start : start + max_words]
        chunks.append(
            Chunk(id=f"{title}#{idx}", title=title, text=" ".join(window), source=source)
        )
        if start + max_words >= len(words):
            break
        start += step
        idx += 1
    return chunks


def documents_from_paths(paths, extensions=(".txt", ".md")) -> list[Document]:
    """Read local text/markdown files into Documents (title = file stem)."""
    docs = []
    for base in paths:
        base = Path(base)
        candidates = [base] if base.is_file() else sorted(base.rglob("*"))
        for f in candidates:
            if f.is_file() and f.suffix.lower() in extensions:
                docs.append(
                    Document(title=f.stem, text=f.read_text(encoding="utf-8"), source=str(f))
                )
    return docs


def write_corpus(chunks, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def read_corpus(path) -> list[Chunk]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(Chunk(**json.loads(line)))
    return chunks
