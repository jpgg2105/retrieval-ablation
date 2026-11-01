# Results: does hybrid search plus reranking pay for itself?

Run: 31 October and 1 November 2026, on a Windows laptop CPU (Intel, 24 threads, PyTorch 2.15 CPU,
sentence-transformers 6.2). Every per-query score and top-10 list is in `results/results.json`;
the tables come from `results/REPORT.md`. Intervals are 95% bootstrap intervals over queries, and
"significant" means the interval of a paired difference excludes zero.

## The short version

- **Production's BM25 tokenizer is its weak point.** ai-document-analyst splits text on
  whitespace (`text.lower().split()`), so `HTTPException,` and `HTTPException` are different
  words. Splitting on word characters lifts BM25's recall@5 on the FastAPI questions from 55% to
  78% (+23 points, significant), and on the 30 questions with an exact term it finds the answer in
  **30 of 30**, more than dense retrieval (26 of 30).
- **With that tokenizer, hybrid adds nothing over dense retrieval.** Hybrid vs dense: +1.7 points
  on FastAPI and −1.7 on SciFact, neither significant, and −3.0 on FiQA (significant). A weak
  keyword list mostly dilutes the dense list in the fusion.
- **The reranker reorders more than it finds**, and it is the expensive part. It raises nDCG@10
  on every dataset (+12 points on FastAPI, significant) while recall@10 moves by 2 points or less.
  It takes 95% or more of query time on CPU, growing in proportion to the number of candidates.
- **Chunks of 512 tokens don't fit the embedder.** MiniLM reads 256 tokens; 90% of the 512-token
  chunks are longer, so the dense side never sees their second half. 256-token chunks scored best
  with the reranker (92% recall@5 vs 85–87% at 512), but on 60 questions that difference is
  within noise.
- **With the tokenizer fixed** (an extra run, added after seeing the results), hybrid + reranker
  is the best system on FastAPI at 90% recall@5, with all 30 exact-term questions answered. Its
  lead over the production pipeline (87%) and over dense + reranker (85%) is not significant on
  60 questions.

## Hypotheses (written before running)

| | Hypothesis | Result |
|---|---|---|
| H1 | Hybrid beats both single retrievers on recall@10 | **Partly.** It always beats BM25 by a wide margin; against dense it is ahead on FastAPI (+3.3) and SciFact (+2.4), not significant, and behind on FiQA (−1.4). |
| H2 | The cross-encoder raises nDCG@10 more than recall@10 | **True** on all three: +12.4 vs +1.7 points (FastAPI), +4.6 vs +2.2 (SciFact), +3.9 vs +2.0 (FiQA). |
| H3 | BM25's advantage over dense is largest on exact-term queries | **False with production's tokenizer**, which never beats dense. **True with the word tokenizer on FastAPI**: BM25 leads dense by 13 points on exact-term questions (significant) and trails by 10 on paraphrased ones. Not on the BEIR sets, where the exact-term split comes from a rule. |
| H4 | The reranker adds the most latency, roughly linear in candidates | **True.** SciFact, one session: 352 ms for 10 candidates, 780 ms for 20, 1,578 ms for 50, 2,982 ms for 100. |

## Main comparison, production settings

Depth 20 per retriever, fused to 20, reranked; recall@5 is what production returns.

| System | FastAPI (60) recall@5 | nDCG@10 | SciFact (300) recall@5 | nDCG@10 | FiQA (648) recall@5 | nDCG@10 |
|---|---|---|---|---|---|---|
| bm25 (production tokenizer) | 0.550 | 0.452 | 0.616 | 0.560 | 0.160 | 0.159 |
| bm25-words | 0.783 | 0.706 | 0.728 | 0.652 | 0.215 | 0.217 |
| dense | 0.767 | 0.663 | 0.738 | 0.645 | 0.367 | 0.369 |
| hybrid (production) | 0.783 | 0.652 | 0.721 | 0.652 | 0.337 | 0.338 |
| dense + rerank | 0.850 | 0.763 | 0.743 | 0.687 | 0.383 | 0.388 |
| **hybrid + rerank (production)** | 0.867 | 0.776 | 0.740 | 0.699 | 0.375 | 0.377 |
| hybrid-words + rerank (extra run) | **0.900** | **0.782** | 0.744 | 0.700 | **0.384** | 0.378 |

FiQA is hard for every system here (financial forum answers, many relevant documents per question);
the ordering between systems is what matters, not the level.

## Where each piece earns its place

**BM25, once tokenized properly, is the exact-term specialist.** FastAPI questions, recall@5:

| Questions | bm25 (prod) | bm25-words | dense | hybrid + rerank (prod) | hybrid-words + rerank |
|---|---|---|---|---|---|
| exact term (30) | 0.600 | **1.000** | 0.867 | 0.933 | **1.000** |
| paraphrased (30) | 0.500 | 0.567 | 0.667 | 0.800 | 0.800 |

**The reranker's quality gain doesn't need many candidates.** Reranking 10 instead of 20 cost
0.017 recall@5 on FastAPI and 0.012 nDCG@10 on SciFact (neither significant) at about half the time.
Going to 50 or 100 candidates added nothing meaningful on either dataset.

**Chunk size** (FastAPI, recall@5; overlap scaled from production's 50 at 512):

| Tokens per chunk | chunks | longer than the embedder's 256 | dense | dense + rerank | hybrid + rerank | bm25-words |
|---|---|---|---|---|---|---|
| 128 | 1,407 | 0% | 0.733 | 0.817 | 0.867 | 0.683 |
| 256 | 687 | 79% | 0.800 | **0.917** | **0.917** | 0.750 |
| 512 (production) | 358 | 90% | 0.767 | 0.850 | 0.867 | 0.783 |
| 1,024 | 204 | 92% | 0.717 | 0.717 | 0.767 | **0.833** |

The two retrievers pull in opposite directions: keyword search likes longer chunks (more words to
match), dense retrieval and the reranker like chunks that fit their input. None of the differences
against 512 is significant on 60 questions, so this is a direction to test, not a result.

## Latency and cost (CPU, one query at a time)

| Stage | FastAPI | SciFact | FiQA |
|---|---|---|---|
| BM25 search (index built once) | 0.4 ms | 0.7 ms | 7 ms |
| Dense: embed the query + similarity | 36 ms | 9 ms | 10 ms |
| RRF fusion | <0.1 ms | <0.1 ms | <0.1 ms |
| Rerank 20 candidates, first run / re-timed | 2,193 / 2,005 ms | 780 / 1,950 ms | 749 / 1,937 ms |

The reranker's absolute time varied by more than 2× between sessions on the same laptop (power
and thermal state), so compare stages within a session: in every session it was over 95% of the
query time. Embedding the corpus once took 17 s (FastAPI), 111 s (SciFact) and 694 s (FiQA, 57,638
posts).

Production's BM25 path rebuilds the whole index from every stored chunk on each query: 24 ms per
query at 358 chunks (p95 39 ms), before Qdrant's scroll, growing with the corpus. Building it once
and updating it on upload removes that cost.

## What I'd change in ai-document-analyst, in order

1. **Tokenize BM25 on word characters**: a one-line change, the largest gain measured here.
2. **Rerank 10 candidates instead of 20**: about half the latency for the same quality on these sets.
3. **Build the BM25 index once** and update it on upload, instead of on every query.
4. **Try 256-token chunks**, which match MiniLM's input; measure on your own documents first.
5. If latency matters more than the last points of nDCG, keep **dense + reranker** and drop the
   fusion: it was within 2 points of the full pipeline everywhere.

## Limits

- 60 hand-written questions over one documentation site: wide intervals, and a corpus with many
  identifiers, which favours keyword search. The BEIR sets cover other kinds of text.
- One embedder and one reranker, both small. Larger models may shift every number here.
- The exact-term split on BEIR comes from a fixed rule, not from labels.
- The `hybrid-words` runs were added after seeing the first results. They test an obvious fix, but
  they were not part of the plan written before the run.
- The questions were written by the author of the pipeline; each answer passage is checked against
  the corpus by `scripts/fastapi_questions.py`.
