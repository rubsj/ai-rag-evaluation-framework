# ADR-003: Embedding Model Comparison, Local vs API

**Date**: 2026-02-13
**Status**: Accepted
**Project**: P2 — RAG Evaluation Benchmarking Framework

## Context

The grid search evaluates 3 embedding models across 5 chunking configurations. The question: does paying for an API embedding model (OpenAI) produce meaningfully better retrieval than free, local models (MiniLM, MPNet)?

| Model | Dimensions | Parameters | Cost | Runs On |
|-------|-----------|------------|------|---------|
| MiniLM (all-MiniLM-L6-v2) | 384 | 22M | Free (local) | CPU/GPU |
| MPNet (all-mpnet-base-v2) | 768 | 109M | Free (local) | CPU/GPU |
| OpenAI (text-embedding-3-small) | 1536 | Unknown | $0.02/1M tokens | API |

Hypothesis going in: OpenAI should win because it has 4x the dimensions of MiniLM and is trained on vastly more data. The interesting question is by how much, and whether the gap is worth the cost and API dependency.

## Decision

**OpenAI text-embedding-3-small is the best model for this corpus**, but I recommend **MiniLM as the default** for development and cost-sensitive work. MPNet underperforms despite being 5x larger than MiniLM.

On Config B (the controlled comparison across models):

| Model | R@1 | R@3 | R@5 | P@1 | MRR@5 |
|-------|-----|-----|-----|-----|-------|
| MiniLM | 0.238 | 0.423 | 0.481 | 0.393 | 0.492 |
| MPNet | 0.146 | 0.347 | 0.467 | 0.268 | 0.398 |
| **OpenAI** | **0.317** | **0.537** | **0.607** | **0.518** | **0.618** |

Averaged across all 5 chunk configs:

| Model | Avg R@5 | Avg MRR@5 | Best Config | Best R@5 |
|-------|---------|-----------|-------------|----------|
| MiniLM | 0.415 | 0.404 | B-minilm | 0.481 |
| MPNet | 0.367 | 0.332 | B-mpnet | 0.467 |
| **OpenAI** | **0.515** | **0.500** | **E-openai** | **0.625** |

OpenAI leads MiniLM by +12.6 pp R@5 on Config B and +10.0 pp averaged across all configs. The top 4 configs in the entire grid search are all OpenAI. MPNet is the worst of the three despite having 5x the parameters of MiniLM (109M vs 22M) and 2x the dimensions (768 vs 384). On Config B, MPNet trails MiniLM by -1.4 pp R@5 and -9.4 pp MRR@5.

OpenAI also benefits most from semantic chunking: E-openai (0.625) is the overall #1 config, while E-minilm (0.452) and E-mpnet (0.415) show smaller gains. All three models beat the BM25 baseline, with MiniLM by +10.0 pp, MPNet by +8.6 pp, and OpenAI by +24.4 pp.

## Alternatives Considered

**OpenAI (best quality)** - Best R@5 (0.607 on Config B), best MRR@5 (0.618), and benefits most from semantic chunking. But it costs $0.02/1M tokens, adds API latency (~200ms), requires internet, and creates vendor lock-in. I selected it for production evaluation runs.

**MPNet (all-mpnet-base-v2)** - Higher dimension (768) than MiniLM and widely recommended in tutorials. But it performed worst of the three on this corpus: 5x MiniLM's parameters for worse results. Bigger does not mean better for this task.

**Larger local models** - Models like `bge-large-en-v1.5` (335M params, 1024 dims) or `e5-large-v2` exist but I didn't test them. The scope was limited to 3 models to keep the grid search tractable at 3x5=15 configs. MiniLM vs MPNet already showed that scaling up local model size didn't help on this corpus. Adding models is straightforward since the infrastructure supports it via the `EmbeddingModel` enum.

## Quantified Validation

OpenAI leads MiniLM by +12.6 pp R@5 on Config B. MPNet trails MiniLM by -1.4 pp despite 5x the parameters. MiniLM gets to 0.481 R@5 at zero cost, while OpenAI reaches 0.607. All three beat BM25, with OpenAI's gap the largest at +24.4 pp. OpenAI amplifies semantic chunking more than local models: E-openai hits 0.625 vs E-minilm at 0.452, a +17.3 pp gap.

## Consequences

MiniLM as the default means zero-cost iteration during development. No API key needed for chunking experiments, test runs, or CI. I used MiniLM for development and switched to OpenAI for production evaluation runs.

MPNet's poor showing may not generalize. On a different corpus (code, legal text), the ranking could change. These conclusions are specific to the Home DIY repair corpus. OpenAI's API dependency means any production RAG system needs a fallback strategy. The 1536-dim OpenAI vectors require 4x the FAISS storage vs 384-dim MiniLM, which is irrelevant at this scale but matters at 1M+ vectors. P3 used MiniLM as the base model for contrastive fine-tuning, with P2's benchmarks establishing it as the quality baseline. (Think of MiniLM as embedded H2 and OpenAI as managed PostgreSQL: you develop locally with the free option and switch to the paid service for production evaluation.)
