"""Chunking, mirroring ai-document-analyst's ``src/ingestion/chunker.py``.

Same algorithm: recursive split on paragraph, line, sentence, word and character boundaries
until every piece fits ``max_tokens`` (counted with tiktoken's cl100k_base), then each chunk
gets the last ``overlap`` tokens of the previous one prepended. Production uses 512 tokens with
an overlap of 50.
"""
from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@lru_cache(maxsize=1)
def _encoding():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


class Tokens:
    """Token counter and codec. cl100k_base when tiktoken is installed, else whitespace words."""

    def __init__(self, use_tiktoken: bool = True) -> None:
        self.exact = False
        if use_tiktoken:
            try:
                _encoding()
                self.exact = True
            except Exception:  # noqa: BLE001 - offline / not installed: fall back, and say so in results
                pass

    def count(self, text: str) -> int:
        return len(_encoding().encode(text)) if self.exact else len(text.split())

    def encode(self, text: str) -> list:
        return _encoding().encode(text) if self.exact else text.split(" ")

    def decode(self, toks: list) -> str:
        return _encoding().decode(toks) if self.exact else " ".join(toks)


def recursive_split(text: str, max_tokens: int, tok: Tokens, separators: list[str] | None = None) -> list[str]:
    seps = list(_SEPARATORS) if separators is None else separators
    if tok.count(text) <= max_tokens:
        return [text]
    if not seps:
        t = tok.encode(text)
        return [tok.decode(t[i : i + max_tokens]) for i in range(0, len(t), max_tokens)]
    sep, rest = seps[0], seps[1:]
    parts = text.split(sep) if sep else list(text)
    chunks: list[str] = []
    cur: list[str] = []
    cur_n = 0
    for part in parts:
        n = tok.count(part)
        if n > max_tokens:
            if cur:
                chunks.append(sep.join(cur))
                cur, cur_n = [], 0
            chunks.extend(recursive_split(part, max_tokens, tok, rest))
            continue
        if cur_n + n > max_tokens and cur:
            chunks.append(sep.join(cur))
            cur, cur_n = [], 0
        cur.append(part)
        cur_n += n
    if cur:
        chunks.append(sep.join(cur))
    return chunks


def add_overlap(pieces: list[str], overlap: int, tok: Tokens) -> list[str]:
    if overlap <= 0 or len(pieces) <= 1:
        return pieces
    out = [pieces[0]]
    for i in range(1, len(pieces)):
        prev = tok.encode(pieces[i - 1])
        out.append(tok.decode(prev[-overlap:]) + " " + pieces[i])
    return out


def chunk_docs(docs: list[dict], size: int, overlap: int, tok: Tokens,
               progress: Callable[[int, int], None] | None = None) -> list[dict]:
    """Split ``{"id", "title", "text"}`` docs into ``{"id", "doc", "text"}`` chunks."""
    out: list[dict] = []
    for n, d in enumerate(docs):
        pieces = add_overlap(recursive_split(d["text"], size, tok), overlap, tok)
        k = 0
        for p in pieces:
            p = p.strip()
            if p:
                out.append({"id": f"{d['id']}#{k}", "doc": d["id"], "text": p})
                k += 1
        if progress:
            progress(n + 1, len(docs))
    return out


def overlap_for(size: int) -> int:
    """Production's ratio: 50 tokens of overlap at 512."""
    return round(size * 50 / 512)
