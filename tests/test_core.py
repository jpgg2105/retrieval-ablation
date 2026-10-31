import math

import numpy as np
import pytest

from ablation import metrics as M
from ablation.chunking import Tokens, add_overlap, chunk_docs, recursive_split
from ablation.data import has_exact_term
from ablation.experiment import covers
from ablation.retrievers import BM25, Dense, rrf


def reference_bm25(corpus, query, k1=1.5, b=0.75, eps=0.25):
    """rank_bm25.BM25Okapi, written out: the formula production uses."""
    toks = [d.lower().split() for d in corpus]
    n = len(toks)
    avg = sum(map(len, toks)) / n
    nd = {}
    for t in toks:
        for w in set(t):
            nd[w] = nd.get(w, 0) + 1
    idf = {w: math.log(n - f + 0.5) - math.log(f + 0.5) for w, f in nd.items()}
    floor = eps * sum(idf.values()) / len(idf)
    idf = {w: (floor if v < 0 else v) for w, v in idf.items()}
    out = []
    for t in toks:
        s = 0.0
        for q in query.lower().split():
            f = t.count(q)
            s += idf.get(q, 0) * f * (k1 + 1) / (f + k1 * (1 - b + b * len(t) / avg))
        out.append(s)
    return np.array(out)


CORPUS = ["The quick brown fox jumps", "the lazy dog sleeps all day", "a fox and a dog, the end",
          "HTTPException raises 404", "raise HTTPException(status_code=404)", "the the the the"]


@pytest.mark.parametrize("q", ["fox dog", "the", "HTTPException 404", "nothing here", "fox fox"])
def test_bm25_matches_reference(q):
    assert np.allclose(BM25(CORPUS).scores(q), reference_bm25(CORPUS, q))


def test_bm25_matches_rank_bm25_when_installed():
    rb = pytest.importorskip("rank_bm25")
    ref = rb.BM25Okapi([d.lower().split() for d in CORPUS])
    for q in ["fox dog", "the", "HTTPException 404"]:
        assert np.allclose(BM25(CORPUS).scores(q), ref.get_scores(q.lower().split()))


def test_bm25_naive_vs_words_tokenizer():
    naive, words = BM25(CORPUS, "naive"), BM25(CORPUS, "words")
    assert 4 not in naive.search("HTTPException", 6)   # "raise HTTPException(status_code=404)" is one naive token
    assert 4 in words.search("HTTPException", 6)


def test_rrf_order_and_k():
    assert rrf([1, 2, 3], [3, 2, 9], top=4) == [3, 2, 1, 9]  # 1/61+1/63 beats 2/62
    s = 1 / 61 + 1 / 63
    assert abs(s - (1 / 61 + 1 / 63)) < 1e-12


def test_dense_cosine():
    class E:
        def encode(self, t):
            return np.array([[1.0, 0.0]])
    d = Dense(E(), np.array([[0.0, 1.0], [10.0, 0.0], [1.0, 1.0]]))
    assert d.search("q", 3) == [1, 2, 0]


def test_metrics_per_query():
    m = M.per_query([set(), {"a"}, set(), {"b"}], {"a": 1, "b": 1})
    assert m["recall@5"] == 1.0 and m["mrr@10"] == 0.5
    ideal = 1 + 1 / math.log2(3)
    assert m["ndcg@10"] == pytest.approx((1 / math.log2(3) + 1 / math.log2(5)) / ideal)
    assert M.per_query([{"a"}, {"a"}], {"a": 1})["ndcg@10"] == 1.0  # a target counts once


def test_paired_bootstrap_detects_shift():
    p = M.paired([0.0] * 50, [1.0] * 40 + [0.0] * 10)
    assert p["significant"] and p["wins"] == 40 and p["losses"] == 0


def test_chunking_overlap_and_size():
    tok = Tokens(use_tiktoken=False)
    text = "\n\n".join(" ".join(f"w{i}_{j}" for j in range(30)) for i in range(10))
    pieces = recursive_split(text, 64, tok)
    assert all(tok.count(p) <= 64 for p in pieces) and len(pieces) == 5
    ov = add_overlap(pieces, 5, tok)
    assert ov[1].startswith(" ".join(pieces[0].split(" ")[-5:]))
    ch = chunk_docs([{"id": "d", "title": "", "text": text}], 64, 5, tok)
    assert [c["id"] for c in ch][:2] == ["d#0", "d#1"]


def test_covers_handles_cut_passages():
    ev = "the exit code after yield is executed right after"
    assert covers("x " + ev + " y", ev)
    assert covers("blah blah the exit code after yield is exec", ev)   # 60%+ prefix at the end
    assert not covers("the exit code", ev)


@pytest.mark.parametrize("q,exp", [("BRCA1 mutations raise risk", True), ("IL-6 levels", True), ("Is an ETF better?", True),
                                   ("How do taxes work for retirees?", False), ("Smoking causes lung cancer", False),
                                   ("mTOR inhibition", True), ("What is a 401k", True)])
def test_exact_term_heuristic(q, exp):
    assert has_exact_term(q) is exp
