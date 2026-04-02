# 🏛️ Elite Operator LLM — Swiss Legal Information Retrieval

> **Kaggle Competition:** [LLM Agentic Legal Information Retrieval](https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval)  
> **Team:** Elite Operator  
> **Repository:** Deep Learning Project for LLM Operator Development and Optimization

---

## 📌 Table of Contents

- [Competition Overview](#-competition-overview)
- [Task Description](#-task-description)
- [Dataset](#-dataset)
- [Evaluation Metric](#-evaluation-metric)
- [Our Approach](#-our-approach)
- [Baseline Model](#-baseline-model)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Results](#-results)
- [Future Work](#-future-work)

---

## 🏆 Competition Overview

This project is our team's submission to the **LLM Agentic Legal Information Retrieval** Kaggle competition. The challenge involves building an intelligent information retrieval system for **Swiss law** — a complex, multilingual legal domain.

The competition tests both NLP capabilities and engineering efficiency, as all inference must run **offline within a 12-hour compute budget**.

---

## 📋 Task Description

Given a legal question written in **English**, the system must retrieve a ranked list of the most relevant Swiss legal sources — including statutes and Federal Court decisions — which are primarily written in **German, French, or Italian**.

Key challenges:
- **Cross-lingual retrieval:** English queries → multilingual legal corpus
- **Exact citation matching:** Predictions must match canonical citation strings precisely
- **Offline inference:** No internet access during execution
- **Compute constraints:** Hard limit of 12 hours total runtime

---

## 📂 Dataset

| File | Description |
|------|-------------|
| `train.csv` | 1,139 public training queries (non-English) with gold-standard citation labels, based on the LEXam dataset |
| `val.csv` | 10 English validation queries with gold citations |
| `test.csv` | 40 hidden English test queries for final submission |
| `laws_de.csv` | 175,933 Swiss federal law snippets in German, keyed by canonical citation string |
| `court_considerations.csv` | Large corpus of Swiss Federal Court decisions *(planned for future iterations)* |

---

## 📊 Evaluation Metric

Submissions are evaluated using **Macro F1 Score** at the citation level.

$$F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

- **True Positive:** A predicted citation that exactly matches a gold citation string
- **Precision:** Fraction of predicted citations that were correct
- **Recall:** Fraction of gold citations that were successfully retrieved
- **Macro F1:** Per-query F1 averaged across all queries

This metric penalizes both over-predicting (too many irrelevant citations) and under-predicting (missing key citations).

---

## 🧠 Our Approach

### Baseline — Multilingual Deep Learning Semantic Search

Since queries are in English but the law corpus is in German, traditional keyword methods like TF-IDF fail due to the language barrier. Our baseline addresses this with a **two-stage hybrid retrieval** pipeline:

#### Stage 1: Deep Learning Semantic Search
We use a **Multilingual Sentence Transformer** (`paraphrase-multilingual-MiniLM-L12-v2`) to encode both the English query and German law snippets into the same dense vector space. Cosine similarity is then computed to retrieve the top-K most semantically relevant articles.

#### Stage 2: Regex Heuristic (Rule-Based Boosting)
We complement neural retrieval with a **regex-based extractor** that detects explicit article mentions in the query (e.g., `Art. 42 OR`) and directly maps them to canonical citation strings. These rule-based hits are merged with the neural results.

**Prediction pipeline:**
```
Query (English)
    │
    ├──► [Multilingual Sentence Transformer] ──► Top-K semantic matches
    │
    └──► [Regex Article Extractor] ──► Direct citation hits
                        │
                        ▼
              Merged & Deduplicated Predictions
```

---

## 📓 Baseline Model

| Property | Details |
|----------|---------|
| **Model** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| **Retrieval** | Cosine similarity over dense embeddings |
| **Heuristic** | Regex pattern matching for article references |
| **Top-K** | 5 citations per query (for submission) |
| **Device** | CUDA (GPU) / CPU fallback |
| **Inference** | Fully offline-compatible |

The notebook `baseline_model.ipynb` contains:
1. Data loading and exploratory analysis
2. Corpus encoding into dense vectors
3. Hybrid prediction function
4. Validation evaluation with F1/Precision/Recall reporting
5. Submission CSV generation
6. Performance visualization dashboard

---

## 📁 Project Structure

```
Elite-Operator-LLM/
├── baseline_model.ipynb      # Main baseline notebook
├── submission.csv            # Generated submission file
├── output.png                # Performance visualization output
├── DATASET/                  # Dataset folder
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── laws_de.csv
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install sentence-transformers torch pandas numpy tqdm matplotlib seaborn
```

### Running the Baseline

1. Clone this repository:
   ```bash
   git clone https://github.com/Eiqbal25/Elite-Operator-LLM.git
   cd Elite-Operator-LLM
   ```

2. Place the competition dataset files into the `DATASET/` folder (or update file paths in the notebook).

3. Open and run `baseline_model.ipynb` end-to-end.

4. The submission file will be saved as `submission.csv`.

> **Note for Kaggle:** The Sentence Transformer model must be downloaded and uploaded as a Kaggle Dataset for offline inference. Replace the model path string in the notebook with your local dataset path (e.g., `/kaggle/input/minilm-model`).

---

## 📈 Results

Performance on the 10-query validation set:

| Metric | Score |
|--------|-------|
| Precision | *(see notebook output)* |
| Recall | *(see notebook output)* |
| **Macro F1** | *(see notebook output)* |

Visual performance reports (bar chart + per-query breakdown + success rate pie chart) are generated at the end of the notebook.

---

## 🔮 Future Work

- [ ] **Expand corpus coverage** — integrate `court_considerations.csv` (Federal Court decisions)
- [ ] **Reranking** — add a cross-encoder reranker for higher precision
- [ ] **Larger multilingual models** — experiment with `multilingual-e5-large` or `LaBSE`
- [ ] **Query translation** — translate English queries to German/French before retrieval
- [ ] **Ensemble strategies** — combine multiple retrievers for improved recall
- [ ] **Fine-tuning** — domain-adapt the embedding model on Swiss legal text pairs

---

## 👥 Team

**Elite Operator** — competing in the Kaggle LLM Operator Development and Optimization competition.

---

## 📄 License

This project is for academic and competition purposes. Dataset usage is subject to [Kaggle competition rules](https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/rules).
