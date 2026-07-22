"""Turn raw MediaWiki markup (wikitext) into clean plain text.

This is intentionally a pragmatic cleaner, not a full wikitext parser: it strips
the constructs that pollute a training/retrieval corpus (templates, refs,
tables, markup) while keeping the human-readable prose and link labels.
"""
from __future__ import annotations

import re

_REF = re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
_REF_SELF = re.compile(r"<ref[^>]*/>", re.IGNORECASE)
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_HEADING = re.compile(r"^=+\s*(.*?)\s*=+\s*$", re.MULTILINE)
_BOLD_ITALIC = re.compile(r"'{2,5}")
_LIST = re.compile(r"^[\*#:;]+\s*", re.MULTILINE)


def _strip_braced(text: str, open_s: str, close_s: str) -> str:
    """Remove balanced {{...}} / {|...|} spans, handling nesting."""
    out, depth, i, n = [], 0, 0, len(text)
    olen, clen = len(open_s), len(close_s)
    while i < n:
        if text[i : i + olen] == open_s:
            depth += 1
            i += olen
        elif text[i : i + clen] == close_s and depth > 0:
            depth -= 1
            i += clen
        else:
            if depth == 0:
                out.append(text[i])
            i += 1
    return "".join(out)


def _resolve_links(text: str) -> str:
    """[[target|label]] -> label ; [[target]] -> target ; drop File:/Image:."""
    def repl(match):
        inner = match.group(1)
        if inner.split(":", 1)[0].lower() in {"file", "image", "category"}:
            return ""
        return inner.split("|")[-1]

    return re.sub(r"\[\[([^\[\]]+)\]\]", repl, text)


def clean_wikitext(raw: str) -> str:
    text = _COMMENT.sub("", raw)
    text = _REF.sub("", text)
    text = _REF_SELF.sub("", text)
    text = _strip_braced(text, "{{", "}}")
    text = _strip_braced(text, "{|", "|}")
    text = _resolve_links(text)
    text = _HEADING.sub(r"\1.", text)
    text = _BOLD_ITALIC.sub("", text)
    text = _LIST.sub("", text)
    text = _TAG.sub("", text)
    text = re.sub(r"\[\S+\s+([^\]]+)\]", r"\1", text)  # [http://x label] -> label
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
