"""Turns results/results.json into results/REPORT.md: tables only, the interpretation is written by hand."""
from __future__ import annotations

import json
from pathlib import Path

from . import metrics as M
from .experiment import SYSTEMS

MAIN = [("fastapi-512", "FastAPI docs (60 questions, 512-token chunks)"), ("scifact", "BEIR SciFact"), ("fiqa", "BEIR FiQA")]
KEYS = ["recall@5", "recall@10", "ndcg@10", "mrr@10"]
PAIRS = [("bm25", "hybrid"), ("dense", "hybrid"), ("hybrid", "hybrid-rerank"), ("dense-rerank", "hybrid-rerank"),
         ("dense", "dense-rerank"), ("bm25", "bm25-words")]


def col(run: dict, s: str, k: str) -> list[float]:
    return [p[k] for p in run["per_query"][s]]


def fmt_ci(xs: list[float]) -> str:
    lo, hi = M.bootstrap(xs)
    return f"{M.mean(xs):.3f} [{lo:.2f}, {hi:.2f}]"


def ms(x: float) -> str:
    return f"{x * 1000:.1f}"


def write_report(path: Path) -> Path:
    res = json.loads(Path(path).read_text(encoding="utf-8"))
    runs = res.get("runs", {})
    L = ["# Retrieval ablation: results", "", f"Environment: `{json.dumps(res.get('environment', {}))}`", ""]
    if res.get("environment", {}).get("fake_models"):
        L += ["> **Dry run with fake models. These numbers mean nothing.**", ""]

    for key, title in MAIN:
        if key not in runs:
            continue
        r = runs[key]
        L += [f"## {title}", "", f"{r['n_queries']} queries, {r['n_units']:,} indexed units, candidate depth {r['depth']}."
              + (f" Skipped (evidence not coverable): {r['skipped_queries']}" if r["skipped_queries"] else ""), "",
              "| System | recall@5 [95% CI] | recall@10 | nDCG@10 [95% CI] | MRR@10 |", "|---|---|---|---|---|"]
        for s in SYSTEMS:
            L.append(f"| {s} | {fmt_ci(col(r, s, 'recall@5'))} | {M.mean(col(r, s, 'recall@10')):.3f} | "
                     f"{fmt_ci(col(r, s, 'ndcg@10'))} | {M.mean(col(r, s, 'mrr@10')):.3f} |")
        L += ["", "Paired differences (same queries), b minus a:", "",
              "| a → b | Δ recall@5 [95% CI] | Δ nDCG@10 [95% CI] | wins / losses (nDCG) |", "|---|---|---|---|"]
        for a, b in PAIRS:
            p5 = M.paired(col(r, a, "recall@5"), col(r, b, "recall@5"))
            pn = M.paired(col(r, a, "ndcg@10"), col(r, b, "ndcg@10"))
            L.append(f"| {a} → {b} | {p5['diff']:+.3f} [{p5['ci'][0]:+.2f}, {p5['ci'][1]:+.2f}]{' *' if p5['significant'] else ''} | "
                     f"{pn['diff']:+.3f} [{pn['ci'][0]:+.2f}, {pn['ci'][1]:+.2f}]{' *' if pn['significant'] else ''} | {pn['wins']} / {pn['losses']} |")
        L += ["", "\\* interval excludes zero.", ""]
        kinds = sorted(set(r["kinds"]))
        if len(kinds) > 1:
            L += ["By query type (H3): dense minus BM25, and hybrid minus dense, on recall@5:", "",
                  "| Type | n | bm25 | dense | hybrid | hybrid-rerank | dense − bm25 [95% CI] |", "|---|---|---|---|---|---|---|"]
            for kd in kinds:
                idx = [i for i, x in enumerate(r["kinds"]) if x == kd]
                sel = {s: [col(r, s, "recall@5")[i] for i in idx] for s in SYSTEMS}
                p = M.paired(sel["bm25"], sel["dense"])
                L.append(f"| {kd} | {len(idx)} | {M.mean(sel['bm25']):.3f} | {M.mean(sel['dense']):.3f} | {M.mean(sel['hybrid']):.3f} | "
                         f"{M.mean(sel['hybrid-rerank']):.3f} | {p['diff']:+.3f} [{p['ci'][0]:+.2f}, {p['ci'][1]:+.2f}] |")
            L.append("")
        lat = r["latency_s"]
        L += ["Latency per query, one query at a time (ms, p50 / p95): " + ", ".join(
              f"{k} {ms(v['p50'])} / {ms(v['p95'])}" for k, v in lat.items()) + ".",
              f"Index: BM25 build {r['index_s']['bm25_build']:.2f} s; embedding the corpus {r['index_s']['embed_corpus']:.1f} s "
              f"({ms(r['index_s']['embed_per_unit'])} ms per unit).", ""]
        if r.get("embedder_truncation"):
            t = r["embedder_truncation"]
            L += [f"Embedder input limit {t['max_seq_length']} tokens: {t['share_truncated']:.0%} of units are longer and get truncated "
                  f"(median {t['median_tokens']:.0f} tokens).", ""]

    sweep = [k for k in ("fastapi-128", "fastapi-256", "fastapi-512", "fastapi-1024") if k in runs]
    if len(sweep) > 1:
        L += ["## Chunk size (FastAPI docs)", "", "recall@5 by system; overlap scales with size (50 at 512).", "",
              "| Chunk tokens | chunks | " + " | ".join(SYSTEMS) + " | truncated by embedder |", "|---" * (len(SYSTEMS) + 3) + "|"]
        for k in sweep:
            r = runs[k]
            tr = r.get("embedder_truncation") or {}
            L.append(f"| {k.split('-')[1]} | {r['n_units']:,} | " + " | ".join(f"{M.mean(col(r, s, 'recall@5')):.3f}" for s in SYSTEMS)
                     + f" | {tr.get('share_truncated', float('nan')):.0%} |")
        L.append("")

    for key, base in (("depth-scifact", "scifact"), ("depth-fastapi-512", "fastapi-512")):
        if key not in runs:
            continue
        d = dict(runs[key])
        rows = []
        if base in runs:
            rows.append(("20", [{"recall@5": a, "ndcg@10": b} for a, b in zip(col(runs[base], "hybrid-rerank", "recall@5"),
                                                                             col(runs[base], "hybrid-rerank", "ndcg@10"))],
                         runs[base]["latency_s"]["rerank"]))
        rows += [(n, v["per_query"], v["rerank_s"]) for n, v in d.items() if n != "wall_s"]
        rows.sort(key=lambda x: int(x[0]))
        L += [f"## Rerank depth ({base})", "", "| Candidates | recall@5 | nDCG@10 | rerank p50 ms | p95 ms |", "|---|---|---|---|---|"]
        for n, per, lt in rows:
            L.append(f"| {n} | {M.mean([p['recall@5'] for p in per]):.3f} | {M.mean([p['ndcg@10'] for p in per]):.3f} | {ms(lt['p50'])} | {ms(lt['p95'])} |")
        L.append("")
    if "bm25-prod" in runs:
        b = runs["bm25-prod"]
        L += ["## Production BM25", "", f"Rebuilding the index on every query ({b['impl']}), 512-token chunks: p50 {ms(b['p50'])} ms, "
              f"p95 {ms(b['p95'])} ms. {b['note']}.", ""]
    out = Path(path).with_name("REPORT.md")
    out.write_text("\n".join(L), encoding="utf-8")
    return out
