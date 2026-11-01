# Retrieval ablation: does hybrid search plus reranking pay for itself?

> Status: **run on 31 October 2026. Full write-up in [RESULTS.md](RESULTS.md).** The hypotheses
> below were written before the run and are not edited afterwards.

## Headline result

- Production's BM25 splits on whitespace, so punctuation sticks to words. Tokenizing on word
  characters lifted its recall@5 from 55% to 78% on the FastAPI questions, and to 30 of 30 on
  questions with an exact term.
- With the production tokenizer, hybrid search added nothing significant over dense retrieval,
  and on FiQA it was 3 points worse.
- The cross-encoder improved ranking (nDCG@10 +4 to +12 points) far more than recall, and took over
  95% of query time on CPU; reranking 10 candidates instead of 20 halved that at the same quality.
- With the tokenizer fixed, hybrid + reranker reached 90% recall@5 on FastAPI (production: 87%),
  a lead that 60 questions can't confirm.

[ai-document-analyst](https://github.com/jpgg2105/ai-document-analyst) retrieves with three pieces:
BM25 keyword search, dense vectors (`all-MiniLM-L6-v2`), and a cross-encoder reranker
(`ms-marco-MiniLM-L-6-v2`), with the BM25 and dense lists merged by Reciprocal Rank Fusion (k = 60).
Its README says each piece helps. This project measures how much each one actually adds, and what
it costs in latency.

## Questions

1. How much does each stage add to retrieval quality: BM25 alone, dense alone, hybrid (RRF), and
   hybrid + cross-encoder?
2. Where does BM25 earn its place? Expectation: on queries with exact terms (codes, error strings,
   names), less so on paraphrased questions.
3. What does the reranker cost? Added latency per query on a laptop CPU, against the quality it adds.
4. How sensitive is all of it to chunk size?

## Hypotheses (written before running)

- H1: hybrid beats both single retrievers on recall@10.
- H2: the cross-encoder raises nDCG@10 more than recall@10 (it reorders more than it finds).
- H3: BM25's advantage over dense retrieval is largest on the exact-term query set.
- H4: the reranker adds the most latency of any stage, roughly linear in the number of candidates.

## Setup

**Data**
- Two public retrieval benchmarks with human relevance labels, from BEIR (Thakur et al., 2021):
  SciFact (300 scientific claims, 5,183 abstracts) and FiQA (648 financial questions, 57,638 posts).
  BEIR queries are split into "exact" and "other" by a rule fixed before running
  (`has_exact_term` in `src/ablation/data.py`: a token with a digit, or with two or more capitals).
- 60 questions over FastAPI's documentation (116 pages, MIT licence, pinned to commit `45cb225`),
  written for this project: 30 contain an exact term (a parameter name, a header, a CLI flag, a
  status code) and 30 describe the need in other words. Each is labelled with a short verbatim
  passage that answers it, and a chunk counts as relevant when it contains the passage, so the
  labels hold for any chunk size. `scripts/fastapi_questions.py` checks every passage against the
  corpus.

**Systems compared.** Every system mirrors ai-document-analyst's code, in memory (no Qdrant), with
its production settings: 512-token chunks with 50 tokens of overlap, 20 candidates from each
retriever, fused to 20, reranked.

| Name | Pipeline | Mirrors |
| --- | --- | --- |
| bm25 | BM25 over `text.lower().split()` tokens, as production does | `retrieval/bm25_search.py` |
| bm25-words | the same, splitting on word characters (extra) | |
| dense | MiniLM embeddings, cosine similarity | `ingestion/embedder.py`, `retrieval/vector_search.py` |
| hybrid | dense + BM25 merged with RRF (k = 60) | `retrieval/fusion.py` |
| dense-rerank | dense top 20, reranked by the cross-encoder (extra) | |
| hybrid-rerank | hybrid top 20, reranked by the cross-encoder: the production path | `api/routes/query.py` |

**Metrics**
- recall@5 (production returns 5 chunks), recall@10, nDCG@10, MRR@10, with 95% bootstrap intervals
- paired differences between systems on the same queries
- latency per query: p50 and p95 for each stage, on CPU, one query at a time
- index build time; the share of chunks longer than the embedder's 256-token input limit

**Additional runs**
- chunk size 128 / 256 / 512 / 1024 tokens on the FastAPI set
- rerank depth 10 / 20 / 50 / 100 candidates on SciFact and FastAPI (quality and latency, for H4)
- production's BM25 path, which rebuilds the index from every stored chunk on each query
- added after the first run, and labelled as such in the results: hybrid built on the
  word-tokenized BM25 (`hybrid-words`, with and without the reranker), because the first run
  showed production's whitespace tokenizer was BM25's weak point

### Changes to the plan, made before running

The first draft of this README planned a hand-made set of about 50 questions, a 256/512/1024
sweep and reranking the top 50. The final setup uses 60 questions, adds 128-token chunks, keeps
production's depth of 20 for the main comparison (with 50 inside the depth sweep), and adds two
extra systems (bm25-words, dense-rerank) to separate the effect of tokenization and of BM25 once a
reranker is present. Hypotheses and metrics are unchanged.

## Run it

Windows, no API key, CPU only:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run.ps1
```

Any OS:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]" && pytest -q
python -m ablation.cli run --fake-models --only fastapi   # 5-second dry run, meaningless numbers
python -m ablation.cli run                                # the real thing
```

Results go to `results/results.json` (every per-query score and top-10 list, so any number can be
recomputed) and `results/REPORT.md` (the tables).

## What makes the result interesting either way

"The reranker added 300 ms for +2 points of recall" is as useful to a team as "BM25 rescued a
third of the exact-term queries". The point is the measured trade-off, not a win.

## References

- Robertson & Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond.
- Cormack, Clarke & Buettcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods. SIGIR.
- Nogueira & Cho (2019). Passage Re-ranking with BERT. arXiv:1901.04085.
- Thakur et al. (2021). BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. NeurIPS Datasets and Benchmarks.
- Reimers & Gurevych (2019). Sentence-BERT. EMNLP.
- FastAPI documentation, © 2018 Sebastián Ramírez, MIT licence (`data/fastapi/LICENSE`).
