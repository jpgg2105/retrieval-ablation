# Retrieval ablation: results

Environment: `{"python": "3.11.9", "platform": "Windows-10-10.0.26200-SP0", "processor": "Intel64 Family 6 Model 183 Stepping 1, GenuineIntel", "cpu_count": 24, "device": "cpu", "fake_models": false, "torch": "2.15.0+cpu", "sentence_transformers": "6.2.0", "torch_threads": 16, "tiktoken": true}`

## FastAPI docs (60 questions, 512-token chunks)

60 queries, 358 indexed units, candidate depth 20.

| System | recall@5 [95% CI] | recall@10 | nDCG@10 [95% CI] | MRR@10 |
|---|---|---|---|---|
| bm25 | 0.550 [0.43, 0.68] | 0.617 | 0.452 [0.35, 0.56] | 0.400 |
| bm25-words | 0.783 [0.67, 0.88] | 0.833 | 0.706 [0.60, 0.80] | 0.665 |
| dense | 0.767 [0.65, 0.87] | 0.883 | 0.663 [0.57, 0.75] | 0.594 |
| hybrid | 0.783 [0.67, 0.88] | 0.917 | 0.652 [0.57, 0.74] | 0.569 |
| dense-rerank | 0.850 [0.75, 0.93] | 0.917 | 0.763 [0.68, 0.85] | 0.714 |
| hybrid-rerank | 0.867 [0.78, 0.95] | 0.933 | 0.776 [0.69, 0.85] | 0.725 |

Paired differences (same queries), b minus a:

| a → b | Δ recall@5 [95% CI] | Δ nDCG@10 [95% CI] | wins / losses (nDCG) |
|---|---|---|---|
| bm25 → hybrid | +0.233 [+0.12, +0.35] * | +0.200 [+0.12, +0.28] * | 32 / 7 |
| dense → hybrid | +0.017 [-0.10, +0.13] | -0.012 [-0.09, +0.07] | 20 / 15 |
| hybrid → hybrid-rerank | +0.083 [-0.02, +0.20] | +0.124 [+0.05, +0.20] * | 23 / 11 |
| dense-rerank → hybrid-rerank | +0.017 [-0.03, +0.08] | +0.012 [-0.00, +0.04] | 5 / 3 |
| dense → dense-rerank | +0.083 [-0.02, +0.18] | +0.100 [+0.02, +0.18] * | 20 / 8 |
| bm25 → bm25-words | +0.233 [+0.13, +0.33] * | +0.253 [+0.15, +0.35] * | 27 / 6 |

\* interval excludes zero.

By query type (H3): dense minus BM25, and hybrid minus dense, on recall@5:

| Type | n | bm25 | dense | hybrid | hybrid-rerank | dense − bm25 [95% CI] |
|---|---|---|---|---|---|---|
| exact | 30 | 0.600 | 0.867 | 0.833 | 0.933 | +0.267 [+0.07, +0.47] |
| paraphrase | 30 | 0.500 | 0.667 | 0.733 | 0.800 | +0.167 [-0.07, +0.40] |

Latency per query, one query at a time (ms, p50 / p95): bm25 0.4 / 0.6, dense 36.1 / 49.5, fuse 0.1 / 0.1, rerank 2193.4 / 2911.2.
Index: BM25 build 0.14 s; embedding the corpus 17.1 s (47.7 ms per unit).

Embedder input limit 256 tokens: 90% of units are longer and get truncated (median 596 tokens).

## BEIR SciFact

300 queries, 5,183 indexed units, candidate depth 20.

| System | recall@5 [95% CI] | recall@10 | nDCG@10 [95% CI] | MRR@10 |
|---|---|---|---|---|
| bm25 | 0.616 [0.56, 0.67] | 0.686 | 0.560 [0.51, 0.61] | 0.524 |
| bm25-words | 0.728 [0.68, 0.78] | 0.774 | 0.652 [0.61, 0.70] | 0.619 |
| dense | 0.738 [0.69, 0.79] | 0.783 | 0.645 [0.60, 0.69] | 0.605 |
| hybrid | 0.721 [0.67, 0.77] | 0.808 | 0.652 [0.61, 0.70] | 0.610 |
| dense-rerank | 0.743 [0.69, 0.79] | 0.811 | 0.687 [0.64, 0.73] | 0.656 |
| hybrid-rerank | 0.740 [0.69, 0.79] | 0.830 | 0.699 [0.65, 0.74] | 0.666 |

Paired differences (same queries), b minus a:

| a → b | Δ recall@5 [95% CI] | Δ nDCG@10 [95% CI] | wins / losses (nDCG) |
|---|---|---|---|
| bm25 → hybrid | +0.105 [+0.07, +0.15] * | +0.093 [+0.07, +0.12] * | 97 / 27 |
| dense → hybrid | -0.017 [-0.05, +0.02] | +0.007 [-0.02, +0.03] | 53 / 58 |
| hybrid → hybrid-rerank | +0.020 [-0.02, +0.06] | +0.046 [+0.02, +0.07] * | 74 / 45 |
| dense-rerank → hybrid-rerank | -0.003 [-0.03, +0.02] | +0.012 [-0.00, +0.03] | 22 / 25 |
| dense → dense-rerank | +0.005 [-0.03, +0.04] | +0.041 [+0.02, +0.07] * | 68 / 41 |
| bm25 → bm25-words | +0.113 [+0.07, +0.16] * | +0.092 [+0.07, +0.12] * | 78 / 22 |

\* interval excludes zero.

By query type (H3): dense minus BM25, and hybrid minus dense, on recall@5:

| Type | n | bm25 | dense | hybrid | hybrid-rerank | dense − bm25 [95% CI] |
|---|---|---|---|---|---|---|
| exact | 172 | 0.638 | 0.752 | 0.746 | 0.718 | +0.114 [+0.05, +0.18] |
| other | 128 | 0.586 | 0.719 | 0.687 | 0.771 | +0.133 [+0.05, +0.21] |

Latency per query, one query at a time (ms, p50 / p95): bm25 0.7 / 1.2, dense 8.5 / 11.0, fuse 0.0 / 0.0, rerank 779.9 / 843.3.
Index: BM25 build 0.44 s; embedding the corpus 110.8 s (21.4 ms per unit).

Embedder input limit 256 tokens: 71% of units are longer and get truncated (median 316 tokens).

## BEIR FiQA

648 queries, 57,638 indexed units, candidate depth 20.

| System | recall@5 [95% CI] | recall@10 | nDCG@10 [95% CI] | MRR@10 |
|---|---|---|---|---|
| bm25 | 0.160 [0.14, 0.18] | 0.204 | 0.159 [0.14, 0.18] | 0.199 |
| bm25-words | 0.215 [0.19, 0.24] | 0.278 | 0.217 [0.19, 0.24] | 0.270 |
| dense | 0.367 [0.34, 0.40] | 0.441 | 0.369 [0.34, 0.40] | 0.445 |
| hybrid | 0.337 [0.31, 0.37] | 0.428 | 0.338 [0.31, 0.36] | 0.407 |
| dense-rerank | 0.383 [0.35, 0.41] | 0.457 | 0.388 [0.36, 0.42] | 0.466 |
| hybrid-rerank | 0.375 [0.35, 0.41] | 0.448 | 0.377 [0.35, 0.40] | 0.454 |

Paired differences (same queries), b minus a:

| a → b | Δ recall@5 [95% CI] | Δ nDCG@10 [95% CI] | wins / losses (nDCG) |
|---|---|---|---|
| bm25 → hybrid | +0.176 [+0.15, +0.20] * | +0.179 [+0.16, +0.20] * | 346 / 34 |
| dense → hybrid | -0.030 [-0.05, -0.01] * | -0.031 [-0.05, -0.02] * | 111 / 248 |
| hybrid → hybrid-rerank | +0.038 [+0.02, +0.06] * | +0.039 [+0.02, +0.05] * | 201 / 143 |
| dense-rerank → hybrid-rerank | -0.008 [-0.02, +0.01] | -0.012 [-0.02, +0.00] | 89 / 142 |
| dense → dense-rerank | +0.016 [-0.01, +0.04] | +0.019 [+0.00, +0.04] * | 196 / 155 |
| bm25 → bm25-words | +0.055 [+0.03, +0.08] * | +0.058 [+0.04, +0.07] * | 167 / 81 |

\* interval excludes zero.

By query type (H3): dense minus BM25, and hybrid minus dense, on recall@5:

| Type | n | bm25 | dense | hybrid | hybrid-rerank | dense − bm25 [95% CI] |
|---|---|---|---|---|---|---|
| exact | 206 | 0.217 | 0.420 | 0.383 | 0.421 | +0.204 [+0.14, +0.27] |
| other | 442 | 0.134 | 0.342 | 0.315 | 0.354 | +0.208 [+0.17, +0.25] |

Latency per query, one query at a time (ms, p50 / p95): bm25 7.1 / 11.3, dense 10.2 / 35.3, fuse 0.0 / 0.0, rerank 748.9 / 1699.5.
Index: BM25 build 2.76 s; embedding the corpus 693.6 s (12.0 ms per unit).

Embedder input limit 256 tokens: 19% of units are longer and get truncated (median 120 tokens).

## Chunk size (FastAPI docs)

recall@5 by system; overlap scales with size (50 at 512).

| Chunk tokens | chunks | bm25 | bm25-words | dense | hybrid | dense-rerank | hybrid-rerank | truncated by embedder |
|---|---|---|---|---|---|---|---|---|
| 128 | 1,407 | 0.417 | 0.683 | 0.733 | 0.733 | 0.817 | 0.867 | 0% |
| 256 | 687 | 0.450 | 0.750 | 0.800 | 0.717 | 0.917 | 0.917 | 79% |
| 512 | 358 | 0.550 | 0.783 | 0.767 | 0.783 | 0.850 | 0.867 | 90% |
| 1024 | 204 | 0.600 | 0.833 | 0.717 | 0.733 | 0.717 | 0.767 | 92% |

## Rerank depth (scifact)

| Candidates | recall@5 | nDCG@10 | rerank p50 ms | p95 ms |
|---|---|---|---|---|
| 10 | 0.739 | 0.687 | 351.8 | 381.5 |
| 20 | 0.740 | 0.699 | 779.9 | 843.3 |
| 50 | 0.745 | 0.695 | 1578.2 | 1775.0 |
| 100 | 0.738 | 0.687 | 2981.9 | 3500.1 |

## Rerank depth (fastapi-512)

| Candidates | recall@5 | nDCG@10 | rerank p50 ms | p95 ms |
|---|---|---|---|---|
| 10 | 0.850 | 0.758 | 385.6 | 410.6 |
| 20 | 0.867 | 0.776 | 2193.4 | 2911.2 |
| 50 | 0.867 | 0.782 | 1732.7 | 2025.2 |
| 100 | 0.867 | 0.767 | 3597.3 | 3710.5 |

## Added after the first run: hybrid on word-tokenized BM25

Not part of the plan: the first run showed production's whitespace tokenizer was BM25's weak point, so the fusion was repeated with the word tokenizer. Same queries, same depth 20.

| Dataset | system | recall@5 | recall@10 | nDCG@10 | Δ recall@5 vs hybrid-rerank [95% CI] | rerank p50 / p95 ms |
|---|---|---|---|---|---|---|
| fastapi-512 | hybrid-words | 0.833 | 0.950 | 0.701 | -0.033 [-0.12, +0.05] |  |
| fastapi-512 | hybrid-words-rerank | 0.900 | 0.967 | 0.782 | +0.033 [-0.03, +0.10] | 2005.3 / 2165.3 |
| scifact | hybrid-words | 0.758 | 0.839 | 0.691 | +0.017 [-0.02, +0.05] |  |
| scifact | hybrid-words-rerank | 0.744 | 0.837 | 0.700 | +0.004 [-0.01, +0.02] | 1950.0 / 2108.6 |
| fiqa | hybrid-words | 0.351 | 0.447 | 0.358 | -0.024 [-0.05, -0.00] * |  |
| fiqa | hybrid-words-rerank | 0.384 | 0.452 | 0.378 | +0.009 [-0.00, +0.02] | 1936.5 / 2106.8 |

## Production BM25

Rebuilding the index on every query (rank_bm25), 512-token chunks: p50 24.1 ms, p95 38.5 ms. index rebuilt per query, as src/retrieval/bm25_search.py does (Qdrant scroll not included).
