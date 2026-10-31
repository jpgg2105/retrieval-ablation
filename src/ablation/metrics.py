"""Ranking metrics with bootstrap intervals.

Relevance is given per query as targets with grades: for BEIR, the judged documents (qrels);
for the FastAPI set, the evidence passages, where a chunk covers a passage when it contains it.
A target counts once, at the first rank that covers it, so recall, nDCG and MRR mean the same
thing whether the unit is a document or a passage. nDCG uses linear gain, as trec_eval does.
"""
from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence


def per_query(covers: Sequence[set], grades: dict, ks: Sequence[int] = (5, 10)) -> dict[str, float]:
    """covers[i] = targets covered by the item at rank i+1; grades = {target: grade > 0}."""
    out: dict[str, float] = {}
    total = len(grades)
    seen: set = set()
    first_hit = 0
    gains: list[float] = []
    for rank, c in enumerate(covers, start=1):
        new = (c & grades.keys()) - seen
        if new and not first_hit:
            first_hit = rank
        seen |= new
        gains.append(sum(grades[t] for t in new))
        for k in ks:
            if rank == k:
                out[f"recall@{k}"] = len(seen) / total
    for k in ks:
        out.setdefault(f"recall@{k}", len(seen) / total)
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains[:10]))
    ideal = sorted(grades.values(), reverse=True)[:10]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    out["ndcg@10"] = dcg / idcg if idcg else 0.0
    out["mrr@10"] = 1.0 / first_hit if 0 < first_hit <= 10 else 0.0
    return out


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap(xs: Sequence[float], n: int = 5000, seed: int = 13, stat: Callable = mean) -> tuple[float, float]:
    if not xs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    m = len(xs)
    vals = sorted(stat([xs[rng.randrange(m)] for _ in range(m)]) for _ in range(n))
    return (vals[int(0.025 * n)], vals[int(0.975 * n) - 1])


def paired(a: Sequence[float], b: Sequence[float], **kw) -> dict:
    """Mean of b - a over the same queries, with a bootstrap interval and win/loss counts."""
    d = [y - x for x, y in zip(a, b)]
    lo, hi = bootstrap(d, **kw)
    return {"diff": mean(d), "ci": [lo, hi], "wins": sum(v > 0 for v in d), "losses": sum(v < 0 for v in d),
            "significant": lo > 0 or hi < 0}


def percentile(xs: Sequence[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = min(len(s) - 1, max(0, math.ceil(p / 100 * len(s)) - 1))
    return s[i]
