"""Build data/fastapi/corpus.jsonl from FastAPI's English docs at a pinned commit.

The docs are MIT-licensed (Copyright (c) 2018 Sebastián Ramírez); see data/fastapi/LICENSE.
Run once; the output is committed so every run uses the same text.

    git clone --filter=blob:none --sparse https://github.com/fastapi/fastapi.git /tmp/fastapi
    git -C /tmp/fastapi checkout 45cb2252c737906f37544f5ba6bee0295fb81468
    git -C /tmp/fastapi sparse-checkout set docs/en/docs
    python scripts/build_fastapi_corpus.py /tmp/fastapi/docs/en/docs
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

COMMIT = "45cb2252c737906f37544f5ba6bee0295fb81468"
SECTIONS = ("tutorial", "advanced", "how-to", "deployment")
TOP_LEVEL = {
    "alternatives.md", "async.md", "benchmarks.md", "environment-variables.md", "fastapi-cli.md",
    "features.md", "history-design-future.md", "python-types.md", "virtual-environments.md",
}

ANCHOR = re.compile(r"\s*\{\s*#[^}]*\}\s*$")
INCLUDE = re.compile(r"^\{\*.*\*\}\s*$")
ADMONITION = re.compile(r"^///\s*\w*(\s*\|.*)?\s*$")


def clean(md: str) -> str:
    out = []
    for line in md.splitlines():
        if INCLUDE.match(line.strip()) or ADMONITION.match(line.strip()):
            continue
        out.append(ANCHOR.sub("", line))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def main(root: str) -> None:
    base = Path(root)
    files = sorted(p for p in base.rglob("*.md")
                   if (p.parent == base and p.name in TOP_LEVEL)
                   or p.relative_to(base).parts[0] in SECTIONS)
    rows = []
    for p in files:
        text = clean(p.read_text(encoding="utf-8"))
        m = re.match(r"#\s+(.+)", text)
        rows.append({"id": p.relative_to(base).as_posix(), "title": m.group(1).strip() if m else p.stem, "text": text})
    dest = Path(__file__).resolve().parents[1] / "data" / "fastapi" / "corpus.jsonl"
    with dest.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} documents, {sum(len(r['text']) for r in rows):,} characters -> {dest} (commit {COMMIT[:7]})")


if __name__ == "__main__":
    main(sys.argv[1])
