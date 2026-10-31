"""Datasets: two BEIR sets with human relevance labels, and the hand-written FastAPI set."""
from __future__ import annotations

import csv
import gzip
import io
import json
import re
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


@dataclass
class Dataset:
    name: str
    docs: list[dict]                       # {"id", "title", "text"}
    queries: list[dict]                    # {"id", "query", "kind"?}
    qrels: dict[str, dict[str, int]] = field(default_factory=dict)       # BEIR: query -> {doc: grade}
    evidence: dict[str, list[str]] = field(default_factory=dict)          # FastAPI: query -> passages
    chunked: bool = False                  # True: docs are split into chunks before indexing


def _read_jsonl(p: Path) -> list[dict]:
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def download_beir(name: str, dest: Path = DATA / "beir") -> Path:
    """Fetch a BEIR dataset once: the official zip, or the Hugging Face mirror if that fails."""
    out = dest / name
    if (out / "qrels" / "test.tsv").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    try:
        print(f"Downloading BEIR {name}...")
        with urllib.request.urlopen(BEIR_URL.format(name=name), timeout=120) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
        for m in z.namelist():
            rel = Path(m).relative_to(name) if m.startswith(name + "/") else None
            if rel and rel.name and not m.endswith("/"):
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
                (out / rel).write_bytes(z.read(m))
    except Exception as e:  # noqa: BLE001
        print(f"  official mirror failed ({e}); trying Hugging Face")
        from huggingface_hub import hf_hub_download

        for fn in ("corpus.jsonl.gz", "queries.jsonl.gz"):
            src = Path(hf_hub_download(f"BeIR/{name}", fn, repo_type="dataset"))
            (out / fn.removesuffix(".gz")).write_bytes(gzip.decompress(src.read_bytes()))
        (out / "qrels").mkdir(exist_ok=True)
        src = Path(hf_hub_download(f"BeIR/{name}-qrels", "test.tsv", repo_type="dataset"))
        (out / "qrels" / "test.tsv").write_bytes(src.read_bytes())
    return out


def load_beir(name: str, dest: Path = DATA / "beir") -> Dataset:
    d = download_beir(name, dest)
    qrels: dict[str, dict[str, int]] = {}
    with open(d / "qrels" / "test.tsv", encoding="utf-8") as f:
        rows = csv.reader(f, delimiter="\t")
        next(rows)
        for qid, did, score in rows:
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)
    docs = [{"id": str(x["_id"]), "title": x.get("title", ""), "text": x.get("text", "")}
            for x in _read_jsonl(d / "corpus.jsonl")]
    qs = [{"id": str(x["_id"]), "query": x["text"]} for x in _read_jsonl(d / "queries.jsonl") if str(x["_id"]) in qrels]
    for q in qs:
        q["kind"] = "exact" if has_exact_term(q["query"]) else "other"
    return Dataset(name, docs, qs, qrels=qrels)


def load_fastapi(root: Path = DATA / "fastapi") -> Dataset:
    docs = _read_jsonl(root / "corpus.jsonl")
    qs = _read_jsonl(root / "questions.jsonl")
    return Dataset("fastapi", docs, [{"id": q["id"], "query": q["query"], "kind": q["kind"]} for q in qs],
                   evidence={q["id"]: q["evidence"] for q in qs}, chunked=True)


_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_./]*")


def has_exact_term(q: str) -> bool:
    """A literal identifier that keyword search can match and an embedding may blur: a token with a
    digit (BRCA1, 401k, p53, 2008), or two or more capitals that aren't just a capitalised first word
    (IL-6, ETF, mTOR, NF-kB). Applied to BEIR queries to split them, decided before running."""
    for i, t in enumerate(_TOKEN.findall(q)):
        if any(c.isdigit() for c in t):
            return True
        caps = sum(c.isupper() for c in t)
        if caps >= 2 or (caps == 1 and i > 0 and not t[0].isupper() and len(t) > 1):
            return True
    return False


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()
