"""Model loading, plus small offline stand-ins used by the tests and the dry run."""
from __future__ import annotations

import hashlib
import re

import numpy as np

EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_models(device: str = "cpu"):
    from sentence_transformers import CrossEncoder, SentenceTransformer

    emb = SentenceTransformer(EMBEDDER, device=device)
    ce = CrossEncoder(RERANKER, device=device)
    return Embedder(emb), ce


class Embedder:
    def __init__(self, model) -> None:
        self.model = model
        self.max_seq_length = getattr(model, "max_seq_length", None)

    def encode(self, texts: list[str], batch_size: int = 64, progress: bool = False) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=progress, convert_to_numpy=True)

    def n_tokens(self, texts: list[str]) -> list[int]:
        tok = self.model.tokenizer
        return [len(x) for x in tok(texts, add_special_tokens=True, truncation=False)["input_ids"]]


_W = re.compile(r"\w+")


class HashEmbedder:
    """Bag-of-words hashed into 256 dimensions: deterministic, offline, good enough to test plumbing."""

    max_seq_length = 256

    def encode(self, texts: list[str], batch_size: int = 64, progress: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in _W.findall(t.lower()):
                h = int(hashlib.md5(w.encode()).hexdigest()[:8], 16)
                out[i, h % 256] += 1.0 if (h >> 9) & 1 else -1.0
        return out

    def n_tokens(self, texts: list[str]) -> list[int]:
        return [len(t.split()) for t in texts]


class OverlapCrossEncoder:
    def predict(self, pairs):
        out = []
        for q, d in pairs:
            qs, ds = set(_W.findall(q.lower())), set(_W.findall(d.lower()))
            out.append(len(qs & ds) / (len(qs) or 1))
        return np.array(out)
