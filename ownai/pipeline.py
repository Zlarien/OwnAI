"""High-level orchestration tying the modules into a runnable pipeline.

These functions are the seams the CLI scripts call. Keeping them here (instead
of inside the scripts) makes the whole flow importable and testable.
"""
from __future__ import annotations

from pathlib import Path

from ownai.data import chunk_document, documents_from_paths, read_corpus, write_corpus
from ownai.retrieval import HybridRetriever


def build_corpus_from_documents(documents, corpus_path, max_words=180, overlap=30):
    """Chunk a list of Documents and persist them as JSONL."""
    chunks = []
    for doc in documents:
        chunks.extend(
            chunk_document(doc.title, doc.text, source=doc.source, max_words=max_words, overlap=overlap)
        )
    write_corpus(chunks, corpus_path)
    return chunks


def build_corpus_from_paths(paths, corpus_path, max_words=180, overlap=30):
    """Ingest local text/markdown files into a chunked corpus."""
    docs = documents_from_paths(paths)
    return build_corpus_from_documents(docs, corpus_path, max_words=max_words, overlap=overlap)


def build_corpus_from_wiki(api_url, category, corpus_path, limit=500, max_words=180, overlap=30):
    """Download a wiki category and turn it into a chunked corpus."""
    from ownai.data.mediawiki import fetch_pages, list_category_pages

    titles = list_category_pages(api_url, category, limit=limit)
    docs = fetch_pages(api_url, titles)
    return build_corpus_from_documents(docs, corpus_path, max_words=max_words, overlap=overlap)


def build_corpus_from_wiki_pages(api_url, titles, corpus_path, max_words=180, overlap=30):
    """Download a hand-picked list of wiki pages -> a clean, focused corpus.

    Prefer this over a whole category when you want tight topical control: a
    category often drags in tangential pages (lists, galleries, cross-game
    references) that pollute the domain.
    """
    from ownai.data.mediawiki import fetch_pages

    docs = fetch_pages(api_url, list(titles))
    return build_corpus_from_documents(docs, corpus_path, max_words=max_words, overlap=overlap)


def build_index_from_corpus(corpus_path, embed_dim=64, w2v_epochs=10, alpha=0.5, seed=0):
    """Load a corpus and build (in memory) the hybrid retriever."""
    chunks = read_corpus(corpus_path)
    retr = HybridRetriever(embed_dim=embed_dim, alpha=alpha, seed=seed)
    retr.index(chunks, w2v_epochs=w2v_epochs)
    return retr


def corpus_to_token_stream(corpus_path, tokenizer):
    """Flatten a corpus into one BPE token stream for language-model training."""
    chunks = read_corpus(corpus_path)
    sep = tokenizer.encode("\n\n")
    stream = []
    for c in chunks:
        stream.extend(tokenizer.encode(c.text))
        stream.extend(sep)
    return stream


def load_domain_config(path):
    """Read a domain YAML config (sources, language, hyperparameters)."""
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
