import json

from ablation import cli


def test_dry_run_fastapi_end_to_end(tmp_path):
    out = tmp_path / "r.json"
    cli.main(["run", "--fake-models", "--only", "fastapi,bm25-prod", "--out", str(out)])
    res = json.loads(out.read_text())
    r = res["runs"]["fastapi-512"]
    assert r["n_queries"] == 60 and not r["skipped_queries"]
    assert set(r["per_query"]) >= {"bm25", "dense", "hybrid", "hybrid-rerank"}
    assert (tmp_path / "REPORT.md").exists()
    # resuming skips finished steps
    cli.main(["run", "--fake-models", "--only", "fastapi", "--out", str(out)])
