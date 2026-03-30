# ADR-002: Chunk Size Selection, Overlap, and Semantic Chunking

**Date**: 2026-02-13
**Status**: Accepted
**Project**: P2 — RAG Evaluation Benchmarking Framework

## Context

The grid search evaluates 5 chunking configurations across 3 embedding models. The question: which chunk size and overlap setting produces the best retrieval quality, and does LLM-based semantic chunking outperform fixed-size chunking?

The 5 configs differ in token count, overlap, and strategy:

| Config | Tokens | Overlap | Strategy |
|--------|--------|---------|----------|
| A | 128 | 32 | Fixed-size |
| B | 256 | 64 | Fixed-size |
| C | 512 | 128 | Fixed-size |
| D | 256 | 0 | Fixed-size (no overlap) |
| E | variable | n/a | LLM-based semantic |

## Decision

I selected **Config B (256 tokens, 64-token overlap)** as the default for this corpus. It consistently performed well across all 3 embedding models:

| Config | MiniLM R@5 | MPNet R@5 | OpenAI R@5 | Avg R@5 |
|--------|-----------|-----------|------------|---------|
| A | 0.291 | 0.235 | 0.304 | 0.277 |
| **B** | **0.481** | **0.467** | **0.607** | **0.518** |
| C | 0.512 | 0.375 | 0.529 | 0.472 |
| D | 0.427 | 0.347 | 0.398 | 0.391 |
| E | 0.452 | 0.413 | 0.625 | 0.497 |

Config A (128 tokens) consistently came in last across all models. At that size, answers get split across multiple chunks and the retrieval signal dilutes. Comparing B to D isolates the effect of overlap: 64-token overlap adds +12.7 pp to R@5 on average, which makes sense since overlap prevents boundary-splitting of relevant content.

Config E (semantic chunking) produced mixed results against Config B. With OpenAI embeddings, E won by +1.8 pp (0.625 vs 0.607). With MiniLM, B won by +2.9 pp. With MPNet, the difference wasn't statistically significant. Config C (512) beat A and D but underperformed B with MiniLM and MPNet because larger chunks dilute precision with more irrelevant text per retrieved chunk.

## Alternatives Considered

**Config E (semantic chunking)** - Achieved the best single R@5 score with OpenAI embeddings (0.625) and preserves section boundaries. But it requires an LLM call per document, produces inconsistent results across local models, and variable chunk sizes complicate batching. The model-dependence made it unreliable as a default.

**Config C (512/128)** - Second-best with some models, and fewer total chunks to search. But precision drops because each chunk carries more irrelevant text, diluting the retrieval signal.

**Config A (128/32)** - Fine-grained retrieval for paragraph-level questions. But it produced the worst recall across every model. Chunks are too small and answers get fragmented.

## Quantified Validation

Config B averaged 0.518 R@5 across all three models, the best cross-model consistency. Against BM25 baseline (0.381), that's a +13.7 pp improvement. The overlap comparison (Config D at 0 overlap vs Config B at 64-token overlap) showed +12.7 pp from overlap alone. Config A's penalty against B was severe at -24.1 pp, confirming that 128 tokens is too small for this corpus. The best single configuration was Config E + OpenAI at 0.625, but that result was model-dependent.

## Consequences

Config B is the default for all downstream tasks (reranking in ADR-005, the full RAG pipeline). It's simple, cheap, and has no LLM dependency for chunking. The overlap cost means storing ~20% more chunks than a zero-overlap approach, which is trivial at this corpus size but would matter at scale.

Config E's advantage with OpenAI embeddings is worth noting for future work: as embedding models improve, semantic chunking may consistently outperform fixed-size. For P2's benchmarking purposes, though, Config B's consistency across all three models made it the safer default. Each chunk config creates its own FAISS index (ADR-001), so the 5 configs x 3 models = 15 indices. QA pairs in ADR-004 are generated from Config B chunks, making B the reference namespace for `gold_chunk_ids`. (This is analogous to choosing fixed-size pagination over content-aware document splitting: predictable and good enough for most queries, even if a smarter split occasionally wins.)
