# 🏛️ Swiss Legal Information Retrieval
### Kaggle — LLM Agentic Legal Information Retrieval
**Leaderboard Score: `0.0700` (Macro F1)**

---

## 📌 Overview

Given a legal question written in **English**, this system retrieves the most relevant Swiss federal law articles and court decisions — primarily written in **German, French, or Italian** — as a semicolon-separated list of exact canonical citation strings.

The core challenge is the **cross-lingual mismatch**: queries are in English but the retrieval corpus is in German. Standard keyword search fails here, so this pipeline combines dense semantic retrieval, lexical BM25 search, and a neural reranker to bridge the language gap.

---

## 🏗️ Pipeline Architecture

```
English Query
      │
      ├──────────────────────────┬────────────────────────────────────────────┐
      │    LAWS RETRIEVAL        │         COURT RETRIEVAL                    │
      │                          │                                            │
      │  BM25 (lexical)          │  Query A: original English query           │
      │       +                  │  Query B: query + top-3 law citations      │
      │  FAISS (dense)           │  Query C: law abbreviations only           │
      │       │                  │           (StPO, OR, ZGB, BV, ...)         │
      │  Weighted RRF            │           max-pool BM25 scores             │
      │  (FAISS ×1.5 / BM25 ×0.5)│                                           │
      └──────────────┬───────────┴──────────────────┬─────────────────────────┘
                     │      Merge + Deduplicate       │
                     └───────────────────────────────┘
                                    │
                           Top-60 candidates
                                    │
                    Chunking + Qwen3-Reranker (MaxSim)
                    (200-word chunks, 50-word overlap)
                                    │
                         Adaptive citation count
                         (score ≥ 0.5, min 10, max 30)
                                    │
                           Final predictions
```

---

## ⚙️ Models Used

| Component | Model | Notes |
|-----------|-------|-------|
| Dense Embedder | `Qwen/Qwen3-Embedding-0.6B` | fp16, max_seq=512, instruction-prefixed queries |
| Neural Reranker | `Qwen/Qwen3-Reranker-0.6B` | Generative yes/no scoring via LM logits |
| Sparse Index | BM25s (`bm25s` library) | German stopwords, disk-cached |
| Vector Index | FAISS `IndexFlatIP` | Inner-product search, disk-cached |

---

## 📂 Dataset

| File | Description |
|------|-------------|
| `train.csv` | Training queries (non-English) with gold citation labels |
| `val.csv` | 10 English validation queries with gold citations |
| `test.csv` | Hidden test set (English queries) |
| `laws_de.csv` | Swiss federal law snippets in German, keyed by canonical citation |
| `court_considerations.csv` | Swiss Federal Court decisions corpus |

Competition page: [llm-agentic-legal-information-retrieval](https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/overview)

---

## 🔑 Key Techniques

### 1. Asymmetrically Weighted RRF
Standard Reciprocal Rank Fusion (equal weights) was replaced with an asymmetric variant that prioritises the dense retriever:

```
score(d) = 0.5 / (60 + rank_BM25) + 1.5 / (60 + rank_FAISS)
```

FAISS is upweighted because semantic embeddings carry stronger cross-lingual signal for English → German retrieval, while BM25 is kept as a supplementary signal for exact abbreviation matching.

### 2. Three-Variant Court Query Expansion
A single English query performs poorly against the German court corpus. Three BM25 queries are issued:
- **Query A** — original English query
- **Query B** — original + top-3 retrieved law citation strings (vocabulary bridge)
- **Query C** — uppercase law abbreviations only (e.g. `SVG ZPO SchKG`) — acts as implicit query translation

### 3. Chunking + MaxSim Reranking
Long court decisions are split into **200-word overlapping chunks** (50-word overlap). Each chunk is scored independently by the Qwen3-Reranker. The **maximum chunk score** (MaxSim) is assigned to the parent document, ensuring relevant passages buried deep in long texts are not discarded.

### 4. Instruction-Prefixed Queries
Qwen3-Embedding requires an instruction prefix to activate its retrieval-optimised representations:
```
Instruct: Given a legal question in English, retrieve relevant Swiss law articles and court decisions written in German.
Query: <query text>
```
Documents are encoded **without** a prefix (Qwen3 convention).

### 5. Adaptive Citation Count
Output size adapts per query rather than using a fixed top-k:
- Include all citations with reranker score **≥ 0.5**
- Minimum **10** citations (safety floor for low-confidence queries)
- Maximum **30** citations (precision cap)

---

## 📊 Results

| Version | Description | Macro F1 |
|---------|-------------|----------|
| Baseline | MiniLM dense retrieval, laws only, fixed top-5 | `0.0661` |
| **v2 (this notebook)** | **Hybrid BM25+FAISS+RRF, Qwen3 reranker, court corpus** | **`0.0700`** |

---

## 🚀 Hyperparameters

```python
TOP_K_RETRIEVAL      = 60      # candidates from BM25 / FAISS before fusion
TOP_K_FINAL          = 10      # minimum citations returned
TEXT_TRUNCATE        = 3500    # max chars per chunk fed to reranker
RRF_K                = 60      # RRF smoothing constant
FAISS_RRF_WEIGHT     = 1.5     # dense retriever weight in RRF
BM25_RRF_WEIGHT      = 0.5     # sparse retriever weight in RRF
LAWS_EXPANSION_TOP_K = 3       # law citations used in court Query B expansion
CHUNK_SIZE           = 200     # words per chunk for MaxSim reranking
CHUNK_OVERLAP        = 50      # overlapping words between chunks
RERANKER_THRESHOLD   = 0.5     # min yes-probability for citation inclusion
MAX_CITATIONS        = 30      # hard cap on output citations
EMBEDDING_BATCH_SIZE = 16
RERANKER_BATCH_SIZE  = 8
```

---

## 💾 Disk Caching

To avoid re-encoding on every run, indexes are serialised to disk:

| Cache | Path | Contents |
|-------|------|----------|
| FAISS laws index | `/kaggle/working/faiss_laws.index` | Dense vectors for laws corpus |
| BM25 laws index | `/kaggle/working/bm25_laws` | Tokenised laws corpus |
| BM25 courts index | `/kaggle/working/bm25_courts` | Tokenised court decisions corpus |

---

## 📦 Requirements

```
sentence-transformers
transformers
bm25s
faiss-cpu
torch
pandas
numpy
tqdm
```

Install:
```bash
pip install sentence_transformers bm25s faiss-cpu
```

---

## 📈 Next Steps

- [ ] Build FAISS dense index for `court_considerations.csv` (biggest recall gap)
- [ ] Fine-tune `Qwen3-Embedding` on `train.csv` with contrastive loss
- [ ] Grid-search reranker threshold & output bounds on `val.csv`
- [ ] Add German query translation as an additional court BM25 query variant
- [ ] Test larger reranker (`Qwen3-Reranker-1.7B`) within compute budget
- [ ] Fine-tune reranker on training (query, citation) pairs

---

## 📄 Evaluation Metric

**Macro F1 Score** at the citation level:
- **Precision** — fraction of predicted citations that are correct
- **Recall** — fraction of gold citations that were retrieved
- F1 is computed per query and averaged across all queries
- Requires **exact string match** of canonical citation format
