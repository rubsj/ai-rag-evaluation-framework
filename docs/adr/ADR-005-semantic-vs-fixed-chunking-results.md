# ADR-005: Semantic vs Fixed-Size Chunking, Experimental Results

**Date**: 2026-02-14
**Status**: Accepted
**Project**: P2 — RAG Evaluation Benchmarking Framework

## Context

The Day 3 grid search evaluated 16 retrieval configurations across 56 synthetic QA pairs. Day 4 added three evaluation layers: Cohere reranking, RAGAS generation evaluation, and LLM-as-Judge. The pipeline degrades at each stage, revealing that retrieval quality does not guarantee generation quality:

| Layer | Metric | E-openai Score |
|-------|--------|----------------|
| Retrieval (Day 3) | R@5 | 0.625 |
| Reranking (Day 4) | R@5 | 0.747 (+19.5%) |
| RAGAS (Day 4) | Faithfulness | 0.511 |
| Judges (Day 4) | Correct | 32.1% |

## Decision

I selected **Config E (semantic chunking) + OpenAI embeddings + Cohere reranking** as the recommended retrieval pipeline. Post-reranking, E-openai achieves Recall@5 = 0.747, the highest across all configurations.

E-openai beat B-openai by 1.8 pp on Recall@5 (0.625 vs 0.607) and wins on per-question-type recall across all answerable categories:

| Question Type | E-openai R@5 | B-openai R@5 | Delta |
|---------------|-------------|-------------|-------|
| Factual (21) | 0.667 | 0.643 | +2.4 pp |
| Multi-hop (19) | 0.645 | 0.632 | +1.3 pp |
| Analytical (12) | 0.653 | 0.639 | +1.4 pp |
| Summarization (3) | 0.300 | 0.278 | +2.2 pp |

B-openai has higher MRR (0.618 vs 0.578), meaning it places the first relevant result higher. Fixed-size chunks are more uniform in scope, so when they match, they match precisely.

RAGAS generation evaluation (GPT-4o-mini) showed the pipeline's weak point: Faithfulness at 0.511 (roughly half of generated claims grounded in context), Answer Relevancy at 0.563, Context Recall at 0.713, and Context Precision at 0.734. Retrieval is strong but generation is losing signal.

The judge evaluation (GPT-4o) uncovered a calibration issue: 21 of 22 refusal answers ("I don't have enough context") were marked as hallucinations. The true hallucination rate on substantive answers is 20/34 (58.8%), not the reported 73.2%. Bloom taxonomy breakdown shows Analyze-level questions account for 68.2% of refusals but only 29.4% of substantive answers.

## Alternatives Considered

**B-openai (fixed 256/64)** - Higher MRR pre-reranking (0.618) with simpler chunking logic and no section-detection dependency. Gets 97% of E's pre-reranking quality. But it gains less from reranking (+9.8% vs +19.5% for E), giving it a lower ceiling. Simpler but leaves performance on the table.

**D-openai (fixed 256/128)** - Highest reranking gain of any config (+26.6%) and decent post-reranking R@5 (0.670). But it has the worst pre-reranking R@5 of the top 3 and produces 22% more chunks than B (717 vs 589) due to high overlap. The reranking gain comes from having more candidates to promote, not from better chunks.

## Quantified Validation

E-openai hit 0.625 R@5 pre-reranking (best single config) and 0.747 post-reranking, a +19.5% improvement. Reranking widened the semantic chunking lead from 1.8 pp to 8.0 pp over B-openai:

| Config | R@5 Before | R@5 After | Improvement |
|--------|-----------|-----------|-------------|
| E-openai | 0.625 | 0.747 | +19.5% |
| B-openai | 0.607 | 0.667 | +9.8% |
| D-openai | 0.529 | 0.670 | +26.6% |

Against BM25 baseline, E-openai post-reranking leads by +36.6 pp R@5. The faithfulness bottleneck at 0.511 confirms that retrieval quality (0.747) does not flow through to generation quality. The judge calibration finding (true hallucination rate 58.8%, not 73.2%) means refusals inflated the count by 14.4 pp.

## Consequences

Semantic chunking preserves document structure (section headers, paragraph boundaries), giving the generator more coherent context passages. Cohere reranking provides a +19.5% recall boost at zero marginal cost on the free tier (1000 calls/month). The 3-layer evaluation establishes baselines for each pipeline stage, and the weakest layer is clearly generation faithfulness at 0.511.

Semantic chunking depends on Markdown header detection. Documents without clear structure (OCR'd PDFs, free-form text) fall back to fixed-size subdivision and lose the advantage. The 39.3% refusal rate on analytical questions suggests the RAG prompt is too conservative, but tuning it risks more hallucination. Judge calibration needs fixing: refusals should not count as hallucinations.

Reranking amplifies semantic chunking's advantage because semantic chunks are more coherent passages for a cross-encoder to score. This interaction (1.8 pp gap pre-reranking, 8.0 pp post-reranking) is the strongest argument for semantic chunking in a reranked pipeline. The faithfulness gap (0.511) identifies the next optimization target as generation prompt engineering, not further retrieval improvement. (This is similar to choosing a ForkJoinPool over a fixed thread pool: semantic chunking adapts to the document's own structure, and the reranker acts as the priority queue on top.)
