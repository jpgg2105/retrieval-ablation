"""The retrieval pipelines compared, each mirroring a piece of ai-document-analyst.

- ``BM25``: ``src/retrieval/bm25_search.py``. Production builds ``rank_bm25.BM25Okapi`` over
  ``text.lower().split()`` tokens and drops results with a score of 0 or less. This class computes
  exactly the same scores (k1=1.5, b=0.75, epsilon=0.25, same IDF floor) from an inverted index, so a
  full benchmark runs in seconds; ``tests/test_bm25_equivalence.py`` checks it against rank_bm25.
  ``tokenizer="words"`` is an extra variant that splits on word characters instead of spaces.
- ``Dense``: ``src/ingestion/embedder.py`` + ``vector_search.py``: all-MiniLM-L6-v2 embeddings,
  cosine similarity (Qdrant's COSINE distance), here in memory.
- ``rrf``: ``src/retrieval/fusion.py``: Reciprocal Rank Fusion with k = 60, dense list first.
- ``Reranker``: ``src/retrieval/reranker.py``: cross-encoder/ms-marco-MiniLM-L-6-v2 over the
  fused candidates.
Production's request path (``src/api/routes/query.py``) is dense top 20 + BM25 top 20, fused to
20, reranked, top 5 returned.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

_WORDS = re.compile(r"\w+")


def tokenize(text: str, mode: str) -> list[str]:
    if mode == "naive":  # production: str.lower().split()
        return text.lower().split()
    if mode == "words":
        return _WORDS.findall(text.lower())
    raise ValueError(mode)


class BM25:
    def __init__(self, texts: list[str], tokenizer: str = "naive", k1: float = 1.5, b: float = 0.75,
                 epsilon: float = 0.25) -> None:
        self.mode, self.k1, self.b = tokenizer, k1, b
        self.n = len(texts)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lens = np.zeros(self.n)
        for i, t in enumerate(texts):
            toks = tokenize(t, tokenizer)
            lens[i] = len(toks)
            for w, f in Counter(toks).items():
                self.postings[w].append((i, f))
        self.avgdl = lens.sum() / self.n
        self.norm = k1 * (1 - b + b * lens / self.avgdl)
        # rank_bm25.BM25Okapi._calc_idf, including its floor for negative IDFs
        idf = {w: math.log(self.n - len(p) + 0.5) - math.log(len(p) + 0.5) for w, p in self.postings.items()}
        floor = epsilon * (sum(idf.values()) / len(idf))
        self.idf = {w: (floor if v < 0 else v) for w, v in idf.items()}
        self._ix = {w: (np.array([d for d, _ in p]), np.array([f for _, f in p], dtype=float))
                    for w, p in self.postings.items()}

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(self.n)
        for w in tokenize(query, self.mode):  # repeated query terms count again, as in rank_bm25
            if w in self._ix:
                d, f = self._ix[w]
                s[d] += self.idf[w] * f * (self.k1 + 1) / (f + self.norm[d])
        return s

    def search(self, query: str, k: int) -> list[int]:
        s = self.scores(query)
        top = np.argsort(-s, kind="stable")[:k]
        return [int(i) for i in top if s[i] > 0]


class Dense:
    def __init__(self, encoder, doc_vecs: np.ndarray) -> None:
        self.encoder = encoder
        self.vecs = _unit(doc_vecs)

    def search(self, query: str, k: int) -> list[int]:
        q = _unit(self.encoder.encode([query]))[0]
        s = self.vecs @ q
        k = min(k, len(s))
        top = np.argpartition(-s, k - 1)[:k]
        return [int(i) for i in top[np.argsort(-s[top], kind="stable")]]


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def rrf(*lists: list[int], k: int = 60, top: int = 20) -> list[int]:
    score: dict[int, float] = {}
    for lst in lists:
        for rank, i in enumerate(lst, start=1):
            score[i] = score.get(i, 0.0) + 1.0 / (k + rank)
    return sorted(score, key=lambda i: score[i], reverse=True)[:top]


class Reranker:
    def __init__(self, cross_encoder, texts: list[str]) -> None:
        self.ce, self.texts = cross_encoder, texts

    def rerank(self, query: str, cands: list[int]) -> list[int]:
        if not cands:
            return []
        s = np.asarray(self.ce.predict([(query, self.texts[i]) for i in cands]))
        order = np.argsort(-s, kind="stable")
        return [cands[j] for j in order]
