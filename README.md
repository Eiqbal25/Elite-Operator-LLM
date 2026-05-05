# Hybrid Legal Retrieval — Kaggle Competition Notebook

**Competition:** [LLM Agentic Legal Information Retrieval](https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval)

**Task:** Given English legal questions, retrieve relevant Swiss law articles and court decisions (written in German).

**Metric:** Macro F1 — per-query F1 between predicted and gold citations, averaged across all queries.

**Current Public Score:** 0.067

---

## Architecture

```
English Query
    │
    ├──► BM25 (laws, German) ──────────┐
    ├──► FAISS dense (laws, German) ───┤── RRF Fusion ──► Law candidates
    │                                   │
    │   ┌───────────────────────────────┘
    │   │
    │   ├──► Query expansion (top-3 law citation strings)
    │   ├──► Law abbreviation extraction (StPO, OR, ZGB, BV...)
    │   │
    │   ├──► BM25 (courts, original query) ────────┐
    │   ├──► BM25 (courts, expanded query) ────────┤── Merge ──► Court candidates
    │   └──► BM25 (courts, abbreviations only) ────┘
    │
    ├──► Merge & deduplicate (laws + courts)
    │
    ├──► Qwen3-Reranker with MaxSim chunking
    │       • Split long docs into 200-word overlapping chunks
    │       • Score each chunk independently
    │       • Assign max chunk score to parent document
    │
    └──► Adaptive citation count
            • Return all with score ≥ 0.5
            • Minimum: TOP_K_FINAL (10)
            • Maximum: 30
```

## Models

| Model | Role | Size | Device |
|-------|------|------|--------|
| Qwen/Qwen3-Embedding-0.6B | Document & query encoding | ~1.2 GB | GPU |
| Qwen/Qwen3-Reranker-0.6B | Relevance scoring (yes/no) | ~1.2 GB | GPU |

Both models loaded simultaneously on a single T4 GPU (15 GB VRAM).

## Datasets

| Dataset | Size | Description |
|---------|------|-------------|
| laws_de.csv | ~175K rows | Swiss federal law articles (German) |
| court_considerations.csv | ~2.47M rows | Swiss court decision excerpts (German) |
| train.csv | Training queries with gold citations |
| val.csv | 10 validation queries with gold citations |
| test.csv | 40 test queries (submission target) |

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| TOP_K_RETRIEVAL | 60 | Candidates retrieved per signal |
| TOP_K_FINAL | 10 | Minimum citations returned per query |
| TEXT_TRUNCATE | 3500 | Max chars for reranker input |
| RRF_K | 60 | Reciprocal rank fusion smoothing |
| LAWS_EXPANSION_TOP_K | 3 | Law citations used for court query expansion |
| CHUNK_SIZE | 200 | Words per reranker chunk |
| CHUNK_OVERLAP | 50 | Overlapping words between chunks |
| EMBEDDING_BATCH_SIZE | 16 | Batch size for corpus encoding |
| RERANKER_BATCH_SIZE | 8 | Batch size for reranker scoring |

## Notebook Structure

| Cell | Section | Description | Runtime |
|------|---------|-------------|---------|
| 1 | Install | `sentence_transformers`, `bm25s`, `faiss-cpu` | ~30s |
| 3 | Config | Imports, paths, hyperparameters, GPU setup | instant |
| 5 | Data Loading | Load train/val/test + laws + court CSVs | ~30s |
| 7 | Text Composition | Build `citation + title + text` strings, lookup dicts | instant |
| 9 | BM25 Index | Build BM25 indexes for laws and courts (German stopwords) | ~5 min |
| 11 | Embedding Model | Load Qwen3-Embedding-0.6B on GPU | ~10s |
| 12 | Encoding Functions | `encode_queries()` (with instruction), `encode_corpus()` (without) | instant |
| 14 | FAISS Index | Encode laws → FAISS (cached to disk after first run) | ~40 min / 2s |
| 16 | Reranker | Load Qwen3-Reranker-0.6B, define MaxSim chunking | ~10s |
| 18 | Retrieval Functions | `search_faiss`, `reciprocal_rank_fusion`, `chunk_text`, `rerank_candidates` | instant |
| 20 | Main Pipeline | `retrieve_citations()` — full pipeline per query | instant |
| 22 | Evaluation | Macro F1 on 10 validation queries + per-query breakdown | ~10 min |
| 24 | Submission | Encode test queries, retrieve, save `submission.csv` | ~30 min |

**Total runtime:** ~50 min (first run) / ~15 min (with FAISS cache)

## Key Improvements Over Original Baseline

| # | Improvement | Impact |
|---|-------------|--------|
| 1 | **Chunking + MaxSim reranking** — splits long documents into 200-word overlapping chunks, scores each, takes max per document | Finds relevant passages buried deep in long court decisions |
| 2 | **Multi-query court retrieval** — 3 BM25 queries instead of 1 (original, expanded, abbreviations) | Better court recall through diverse query formulations |
| 3 | **Adaptive citation count** — returns more when reranker is confident (score ≥ 0.5) | Adapts to queries with many vs few relevant citations |
| 4 | **FAISS disk caching** — saves encoded index to `/kaggle/working/faiss_laws.index` | Reduces restart time from ~40 min to ~2 seconds |

## Caching

| Cache | Path | First Run | Restarts |
|-------|------|-----------|----------|
| FAISS laws index | `/kaggle/working/faiss_laws.index` | ~40 min (encode + save) | ~2 seconds (load) |

## How Qwen3-Embedding Works

Qwen3-Embedding uses **instruction-aware encoding**:

- **Queries** are prefixed with: `"Instruct: {task_instruction}\nQuery: {query_text}"`
- **Documents** are encoded as-is (no prefix)

This tells the model what kind of retrieval task is being performed, improving cross-lingual matching (English queries → German documents).

## How Qwen3-Reranker Works

Qwen3-Reranker is a **causal language model** (not a cross-encoder). It judges relevance by:

1. Formatting query + document into a chat template
2. Running a forward pass
3. Extracting logits for tokens "yes" and "no"
4. Computing `P(yes)` as the relevance score (0-1)

Loaded via `AutoModelForCausalLM`, NOT `CrossEncoder` (which would give random scores).

## Known Limitations

- **Cross-lingual gap**: English BM25 queries match poorly against German corpus. Only shared terms (law abbreviations, article numbers) provide useful BM25 signal.
- **Court recall**: 2.47M court decisions are too large for dense encoding on T4. Courts use BM25-only retrieval.
- **Retrieval ceiling**: Debug shows only 3-6 out of 36-47 gold citations are found in the candidate pool. The reranker can only rank what was retrieved.

## Potential Further Improvements

| Improvement | Difficulty | Expected Impact |
|-------------|-----------|-----------------|
| Query translation EN→DE (Helsinki-NLP/opus-mt-en-de, ~300MB) | Easy | High — German BM25 on German text |
| Dense court encoding (chunked FAISS for 2.47M rows) | Hard (OOM) | High — semantic court retrieval |
| HyDE (generate hypothetical German analysis per query) | Medium | Medium — bridges cross-lingual gap |
| Citation graph (PageRank, co-citation from cross-references) | Medium | Medium — exploits legal citation structure |
| Fine-tuning embedding model on Swiss legal data | Hard | High — domain adaptation |
| Agentic RAG with LLM-guided query reformulation | Hard | High — multi-step retrieval |

## File Structure

```
├── baseline-upgrade-0-067.ipynb    # Main notebook
├── README.md                       # This file
├── submission.csv                  # Output (generated)
└── /kaggle/working/
    └── faiss_laws.index            # Cached FAISS index
```

## Reproduction

1. Upload notebook to Kaggle
2. Ensure GPU (T4) and Internet are enabled
3. Attach competition dataset
4. Run All Cells
5. First run: ~50 minutes. Subsequent runs: ~15 minutes (FAISS cached)
6. Output: `submission.csv`
