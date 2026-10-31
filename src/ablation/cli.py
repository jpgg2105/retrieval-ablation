"""python -m ablation.cli run [--fake-models] [--only fastapi,scifact,fiqa] [--quick]

Steps (each saved to results/results.json as it finishes, and skipped when you run again):
  1. fastapi-512   the production setup: 512-token chunks, depth 20, all systems
  2. fastapi sweep chunk sizes 128 / 256 / 1024
  3. scifact       BEIR SciFact, all systems
  4. fiqa          BEIR FiQA, all systems
  5. depth         rerank depth 10 / 50 / 100 on SciFact and fastapi-512
  6. bm25-prod     latency of production's rebuild-per-query BM25 on fastapi-512
Then results/REPORT.md is written from the JSON.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import experiment as E
from .chunking import Tokens
from .data import ROOT, load_beir, load_fastapi

RESULTS = ROOT / "results"


def _save(res: dict, path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(res), encoding="utf-8")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="ablation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--fake-models", action="store_true", help="offline stand-ins, to test the pipeline")
    r.add_argument("--only", default="", help="comma list of steps: fastapi,sweep,scifact,fiqa,depth,bm25-prod")
    r.add_argument("--device", default="cpu")
    r.add_argument("--out", default=str(RESULTS / "results.json"))
    sub.add_parser("report")
    a = ap.parse_args(argv)

    if a.cmd == "report":
        from .report import write_report
        print(write_report(RESULTS / "results.json"))
        return

    out = Path(a.out)
    res = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    if a.fake_models:
        from .models import HashEmbedder, OverlapCrossEncoder
        emb, ce, tok = HashEmbedder(), OverlapCrossEncoder(), Tokens(use_tiktoken=False)
    else:
        from .models import load_models
        print("Loading models (downloaded once)...")
        emb, ce = load_models(a.device)
        tok = Tokens()
    res["environment"] = E.environment(a.device, a.fake_models)
    res["environment"]["tiktoken"] = tok.exact
    steps = [s for s in (a.only.split(",") if a.only else ["fastapi", "sweep", "scifact", "fiqa", "depth", "bm25-prod"]) if s]
    fa = load_fastapi()
    units = {}

    def fa_units(size: int):
        if size not in units:
            units[size] = E.Units(fa, size, tok)
        return units[size]

    def step(key: str, fn):
        if key in res.get("runs", {}):
            print(f"[skip] {key} (already in {out.name})")
            return
        print(f"[run]  {key}")
        t = time.perf_counter()
        val = fn()
        val["wall_s"] = time.perf_counter() - t
        res.setdefault("runs", {})[key] = val
        _save(res, out)
        print(f"       done in {val['wall_s']:.0f} s")

    if "fastapi" in steps:
        step("fastapi-512", lambda: E.run_systems(fa, fa_units(512), emb, ce, tag="fastapi-512"))
    if "sweep" in steps:
        for size in (128, 256, 1024):
            step(f"fastapi-{size}", lambda s=size: E.run_systems(fa, fa_units(s), emb, ce, tag=f"fastapi-{s}"))
    beir = {}
    for name in ("scifact", "fiqa"):
        if name in steps or (name == "scifact" and "depth" in steps):
            beir[name] = load_beir(name)
        if name in steps:
            step(name, lambda n=name: E.run_systems(beir[n], E.Units(beir[n], None, tok), emb, ce))
    if "depth" in steps:
        step("depth-scifact", lambda: E.depth_sweep(beir["scifact"], E.Units(beir["scifact"], None, tok), emb, ce))
        step("depth-fastapi-512", lambda: E.depth_sweep(fa, fa_units(512), emb, ce, tag="fastapi-512"))
    if "bm25-prod" in steps:
        step("bm25-prod", lambda: E.production_bm25_latency(fa_units(512), fa.queries))

    from .report import write_report
    print(f"\nReport: {write_report(out)}")
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
