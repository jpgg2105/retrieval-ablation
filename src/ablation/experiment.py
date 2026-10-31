"""Runs the experiments and writes results/results.json (per-query data included, so every number
in the report can be recomputed)."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from . import metrics as M
from .chunking import Tokens, chunk_docs, overlap_for
from .data import ROOT, Dataset, norm
from .retrievers import BM25, Dense, Reranker, rrf

SYSTEMS = ["bm25", "bm25-words", "dense", "hybrid", "dense-rerank", "hybrid-rerank"]
CACHE = ROOT / ".cache"


class Units:
    """What gets indexed: whole documents (BEIR) or chunks (FastAPI)."""

    def __init__(self, ds: Dataset, size: int | None, tok: Tokens) -> None:
        self.ds, self.size = ds, size
        if ds.chunked:
            ch = chunk_docs(ds.docs, size, overlap_for(size), tok)
            self.ids = [c["id"] for c in ch]
            self.texts = [c["text"] for c in ch]
        else:
            self.ids = [d["id"] for d in ds.docs]
            self.texts = [(d["title"] + "\n" + d["text"]).strip() for d in ds.docs]
        self._norm = [norm(t) for t in self.texts] if ds.chunked else None

    def grades(self, qid: str) -> dict:
        if self.ds.chunked:
            return {j: 1 for j in range(len(self.ds.evidence[qid]))}
        return self.ds.qrels[qid]

    def covers(self, qid: str, ranked: list[int]) -> list[set]:
        if not self.ds.chunked:
            return [{self.ids[i]} for i in ranked]
        ev = [norm(e) for e in self.ds.evidence[qid]]
        return [{j for j, e in enumerate(ev) if covers(self._norm[i], e)} for i in ranked]

    def coverable(self, qid: str) -> bool:
        ev = [norm(e) for e in self.ds.evidence[qid]] if self.ds.chunked else []
        return all(any(covers(t, e) for t in self._norm) for e in ev)


def covers(chunk: str, ev: str, min_frac: float = 0.6) -> bool:
    """The chunk contains the passage, or 60%+ of it where a chunk boundary cut it."""
    if ev in chunk:
        return True
    for k in range(len(ev) - 1, int(len(ev) * min_frac) - 1, -1):
        if chunk.endswith(ev[:k]) or chunk.startswith(ev[-k:]):
            return True
    return False


def embed_cached(emb, texts: list[str], tag: str) -> tuple[np.ndarray, float]:
    h = hashlib.sha1(("\x00".join(texts) + type(emb).__name__).encode()).hexdigest()[:12]
    p = CACHE / f"emb-{tag}-{h}.npy"
    if p.exists():
        meta = json.loads(p.with_suffix(".json").read_text())
        return np.load(p), meta["seconds"]
    CACHE.mkdir(exist_ok=True)
    t = time.perf_counter()
    v = emb.encode(texts, progress=len(texts) > 2000)
    sec = time.perf_counter() - t
    np.save(p, v)
    p.with_suffix(".json").write_text(json.dumps({"seconds": sec, "n": len(texts)}))
    return v, sec


def run_systems(ds: Dataset, units: Units, emb, ce, depth: int = 20, tag: str = "") -> dict:
    t = time.perf_counter(); bm = BM25(units.texts, "naive"); t_bm = time.perf_counter() - t
    bmw = BM25(units.texts, "words")
    vecs, t_emb = embed_cached(emb, units.texts, tag or ds.name)
    dense, rr = Dense(emb, vecs), Reranker(ce, units.texts)
    qs = [q for q in ds.queries if not ds.chunked or units.coverable(q["id"])]
    # warm up (first calls allocate)
    dense.search(qs[0]["query"], depth); rr.rerank(qs[0]["query"], list(range(min(5, len(units.texts)))))
    per = {s: [] for s in SYSTEMS}
    ranks = {s: [] for s in SYSTEMS}
    lat = {k: [] for k in ("bm25", "dense", "fuse", "rerank")}
    for q in qs:
        text = q["query"]
        t0 = time.perf_counter(); b = bm.search(text, depth)
        t1 = time.perf_counter(); d = dense.search(text, depth)
        t2 = time.perf_counter(); h = rrf(d, b, top=depth)
        t3 = time.perf_counter(); hr = rr.rerank(text, h)
        t4 = time.perf_counter()
        lat["bm25"].append(t1 - t0); lat["dense"].append(t2 - t1); lat["fuse"].append(t3 - t2); lat["rerank"].append(t4 - t3)
        out = {"bm25": b, "bm25-words": bmw.search(text, depth), "dense": d, "hybrid": h,
               "dense-rerank": rr.rerank(text, d), "hybrid-rerank": hr}
        g = units.grades(q["id"])
        for s, lst in out.items():
            per[s].append(M.per_query(units.covers(q["id"], lst[:10]), g))
            ranks[s].append([units.ids[i] for i in lst[:10]])
    trunc = None
    if hasattr(emb, "n_tokens"):
        n = emb.n_tokens(units.texts)
        trunc = {"max_seq_length": emb.max_seq_length,
                 "share_truncated": sum(x > (emb.max_seq_length or 10**9) for x in n) / len(n),
                 "median_tokens": float(np.median(n))}
    return {
        "n_units": len(units.texts), "n_queries": len(qs), "depth": depth,
        "skipped_queries": [q["id"] for q in ds.queries if q not in qs],
        "query_ids": [q["id"] for q in qs], "kinds": [q.get("kind", "") for q in qs],
        "per_query": per, "top10": ranks,
        "latency_s": {k: {"p50": M.percentile(v, 50), "p95": M.percentile(v, 95), "mean": M.mean(v)} for k, v in lat.items()},
        "index_s": {"bm25_build": t_bm, "embed_corpus": t_emb, "embed_per_unit": t_emb / len(units.texts)},
        "embedder_truncation": trunc,
    }


def depth_sweep(ds: Dataset, units: Units, emb, ce, depths=(10, 50, 100), tag: str = "") -> dict:
    bm = BM25(units.texts, "naive")
    vecs, _ = embed_cached(emb, units.texts, tag or ds.name)
    dense, rr = Dense(emb, vecs), Reranker(ce, units.texts)
    qs = [q for q in ds.queries if not ds.chunked or units.coverable(q["id"])]
    out = {}
    for n in depths:
        per, lat = [], []
        for q in qs:
            h = rrf(dense.search(q["query"], n), bm.search(q["query"], n), top=n)
            t = time.perf_counter(); hr = rr.rerank(q["query"], h); lat.append(time.perf_counter() - t)
            per.append(M.per_query(units.covers(q["id"], hr[:10]), units.grades(q["id"])))
        out[str(n)] = {"per_query": per, "rerank_s": {"p50": M.percentile(lat, 50), "p95": M.percentile(lat, 95)}}
        print(f"    depth {n}: recall@5 {M.mean([p['recall@5'] for p in per]):.3f}, rerank p50 {M.percentile(lat, 50)*1000:.0f} ms")
    return out


def production_bm25_latency(units: Units, queries: list[dict]) -> dict:
    """Production rebuilds the BM25 index from every stored chunk on each query. Time that."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        BM25Okapi = None
    lat = []
    for q in queries:
        t = time.perf_counter()
        if BM25Okapi:
            BM25Okapi([x.lower().split() for x in units.texts]).get_scores(q["query"].lower().split())
        else:
            BM25(units.texts, "naive").scores(q["query"])
        lat.append(time.perf_counter() - t)
    return {"impl": "rank_bm25" if BM25Okapi else "local", "p50": M.percentile(lat, 50), "p95": M.percentile(lat, 95),
            "note": "index rebuilt per query, as src/retrieval/bm25_search.py does (Qdrant scroll not included)"}


def environment(device: str, fake: bool) -> dict:
    env = {"python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor(),
           "cpu_count": os.cpu_count(), "device": device, "fake_models": fake}
    try:
        import sentence_transformers
        import torch

        env.update(torch=torch.__version__, sentence_transformers=sentence_transformers.__version__,
                   torch_threads=torch.get_num_threads())
    except ImportError:
        pass
    return env
