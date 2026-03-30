# ADR-001: FAISS over ChromaDB/LanceDB for Benchmarking

**Date**: 2026-02-13
**Status**: Accepted
**Project**: P2 — RAG Evaluation Benchmarking Framework

## Context

P2 is a benchmarking framework evaluating 16 retrieval configurations: 5 chunking strategies × 3 embedding models + a BM25 baseline. Each config requires its own vector index with direct access to raw similarity scores for Recall@K, Precision@K, and MRR@K computation. The corpus runs ~500 to 1200 chunks per config (under 1K vectors), so approximate nearest neighbor, metadata filtering, and server processes are unnecessary overhead.

## Decision

I used **faiss-cpu with IndexFlatIP** (brute-force inner product) as the vector store, wrapped in a thin `FAISSVectorStore` class (~90 lines in `src/vector_store.py`).

`IndexFlatIP` computes exact inner product. Since the embedder L2-normalizes all vectors, inner product equals cosine similarity, with zero approximation error. Brute-force is fast at this scale: under 1ms per query for under 1K vectors. Approximate indices (IVF, HNSW) would add complexity with no benefit below ~10K vectors.

Persistence is two files per index: a `.faiss` binary via FAISS native `write_index`, plus a `.json` sidecar storing the chunk ID list. The sidecar is human-readable for debugging and maps FAISS integer positions back to chunk IDs. Validation on `add()` catches dimension mismatch and ID count mismatch immediately, not silently at search time. The same `search(query, k)` return signature is shared with `BM25Retriever`.

## Alternatives Considered

**ChromaDB** - Built-in persistence and metadata filtering, auto-generates IDs, popular in RAG tutorials. But ChromaDB's abstraction hides raw similarity scores (it returns distances, not similarities), manages its own SQLite/DuckDB storage, and makes it harder to create and load 15 separate collections cleanly. For a benchmarking framework that needs to compare raw retrieval quality across configs, that abstraction layer works against you.

**LanceDB** - Columnar on-disk format, embedded with no server, scales well for larger datasets. But it's overkill for under 1K vectors, adds a columnar format dependency, and abstracts away the index internals I needed to inspect. Over-engineered for the scale I was working at.

## Quantified Validation

I built and loaded 15 separate indices independently with zero conflicts. Query latency on under 1K vectors was under 1ms, so brute-force was faster than ANN index setup overhead at this scale. The `FAISSVectorStore` wrapper came in at ~90 lines of code, fully auditable with no hidden behavior. The two-file persistence pattern (`.faiss` + `.json` sidecar) made it possible to inspect chunk ID mappings when debugging cross-config gold chunk resolution.

## Consequences

Raw similarity scores flow directly into metric computation with no translation layer. Each config's index is a pair of files I can inspect, move, or delete independently.

Chunk ID mapping is manual: I maintain a parallel list and a JSON sidecar file. There's no metadata filtering, which wasn't needed for benchmarking. If this needed to scale to 1M+ vectors, I'd need to switch to IndexIVFFlat or HNSW and add training steps.

P3 reused FAISS IndexFlatIP for embedding similarity evaluation. P4 switched to ChromaDB for production use (metadata filtering, live API), which confirmed that FAISS was the right call for benchmarking but not sufficient for serving. (This is roughly the same tradeoff as raw JDBC vs. Hibernate in Java: you want the raw driver when you need to see every result set, and the managed layer when you need lifecycle and filtering.)
