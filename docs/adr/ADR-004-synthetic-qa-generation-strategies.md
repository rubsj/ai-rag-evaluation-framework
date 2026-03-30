# ADR-004: Synthetic QA Generation, 5 Strategies with Instructor Validation

**Date**: 2026-02-13
**Status**: Accepted
**Project**: P2 — RAG Evaluation Benchmarking Framework

## Context

Evaluating 16 retrieval configurations requires a gold-standard question set where each question has known `gold_chunk_ids` pointing to the chunks that contain the answer. I needed questions covering different cognitive levels (factual, multi-hop, analytical), different retrieval scopes (single chunk through full page), and edge cases like chunk boundary overlap regions. All questions had to be answerable from the corpus with verifiable gold chunk mappings for metric computation. Manually writing 50+ questions for a 589-chunk corpus was impractical.

## Decision

I used **5 complementary generation strategies** to produce 56 synthetic QA pairs from Config B chunks (256 tokens, 64-token overlap). Each strategy targets a different retrieval challenge. All LLM calls use GPT-4o-mini via Instructor for structured output validation.

| # | Strategy | Questions | Gold Chunks/Q | Purpose |
|---|----------|-----------|---------------|---------|
| 1 | Per-Chunk Chain | 24 | 1 | Baseline: can the system find a single chunk? |
| 2 | Multi-Chunk | 10 | 2-4 | Cross-chunk: find semantically related chunks? |
| 3 | Overlap Region | 8 | 2 | Boundary test: content split across chunk boundaries |
| 4 | Hierarchical | 8 | 1-5+ | Scope test: paragraph vs section vs page-level |
| 5 | Academic Pattern | 6 | 1 | Format diversity: definition, comparison, application |

The resulting question types: 21 factual, 19 multi-hop, 12 analytical, 3 summarization, 1 comparative.

All `gold_chunk_ids` are in Config B namespace (e.g., `B_0_42`). When evaluating other configs, gold IDs are mapped via character position overlap in `grid_search.map_gold_chunks()`. I chose Config B as the reference because BM25 uses the same chunks, so BM25 evaluation needs no mapping.

Every LLM call returns a Pydantic model through Instructor, which auto-retries on validation failure (up to 3 attempts). This eliminates JSON parsing errors. Strategy 2 finds related chunks by extracting embeddings from `minilm_B.faiss` via `index.reconstruct_n()`, so no embedding model needs to be loaded at QA generation time. Every Instructor call is cached via `cache.compute_cache_key()`, so regenerating QA pairs costs $0 after the first run. Chunk sampling uses `_sample_diverse_chunks()` to spread selections across all 3 documents and sample from beginning, middle, and end positions within each.

## Alternatives Considered

**Single-strategy (per-chunk only)** - Simple, one prompt template, easy to scale. But all questions would be single-chunk factual, completely missing multi-hop and boundary cases. A question set that only tests one retrieval pattern gives misleading confidence.

**Manual question writing** - Highest quality with perfect gold mappings. But impractical for 50+ questions, doesn't scale, and introduces subjective bias in what gets tested.

**RAGAS-style automatic generation** - Battle-tested library that handles multi-hop generation. But it's a black box: I couldn't control the strategy mix, couldn't easily verify gold chunk mappings, and it adds a dependency. For a benchmarking framework, I needed to know exactly what each question was testing.

56 questions is sufficient for comparing 16 configs. Statistical significance requires ~30+ observations per group, and 56 questions across 16 configs gives 896 data points. The top-3 ranking would not change with 100 or 200 questions on this corpus.

## Quantified Validation

I generated 56 QA pairs across 5 strategies, exceeding the 50-question threshold. Instructor validation passed on 100% of generated questions with zero rejections. The question type distribution is 21 factual, 19 multi-hop, 12 analytical, 3 summarization, and 1 comparative. Chunk coverage is 12.4% (73 of 589 chunks), which is a known gap. Total generation cost was ~$0.08 for the first run and $0.00 thereafter thanks to caching. Across the grid search, that's 896 data points (56 questions x 16 configs).

## Consequences

Gold chunk IDs are deterministic since each strategy explicitly assigns them from the source chunks used in generation. No post-hoc annotation needed. The 5-strategy mix catches retrieval weaknesses that a single-strategy evaluation would miss. Caching means QA pairs are generated once and reused across all grid search runs.

12.4% chunk coverage means 87.6% of chunks have no gold question. If a retrieval bug only affects uncovered chunks, I won't detect it. Strategy 2 (multi-chunk) depends on pre-built FAISS indices existing (ADR-001), which creates an ordering dependency in the pipeline. The comparative question type has only 1 question, too few to draw per-type conclusions.

P4 adopted multi-strategy generation with controlled fit levels instead of question strategies. The pattern of generating from a reference config and evaluating across all configs turned out to be reusable for RAG benchmarking generally. (This is similar to building a JUnit suite with unit, integration, and boundary tests rather than relying on a single test category: coverage diversity matters as much as volume.)
