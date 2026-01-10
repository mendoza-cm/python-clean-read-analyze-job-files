# Python Job File Analysis & Text Search

This project demonstrates a full end-to-end pipeline for cleaning, profiling, and analyzing heterogeneous job posting data scraped from multiple employment websites.  
It includes exploratory analysis, data standardization, and an evolving text-based job search/retrieval engine.

> **Note:** Only cleaned CSV outputs and analysis code are included.  
> The original web scraping logic is intentionally excluded.

---

## Project Goals

- Ingest and standardize inconsistent job posting CSVs  
- Infer types, profile missingness/uniqueness, and group columns logically  
- Prepare rich text representations for NLP/ML workflows  
- Build and progressively improve text-based job matching  
  (starting with lexical → moving toward semantic/hybrid approaches)  
- Create a modular foundation for future job-matching experiments  

---

## Repository Structure
````
python-clean-read-analyze-job-files/
├── data_management.py      # Ingestion, cleaning, profiling, text preparation
├── ml_text_search.py       # Similarity/ranking methods (TF-IDF, BM25, embeddings, hybrid...)
├── main.py                 # Example end-to-end workflow
├── Data/                   # Raw scraped CSVs (inputs)
├── Results/                # Outputs: plots, summaries, experiment logs
├── docs/                   # Detailed method explanations
│   ├── tfidf_cosine_similarity.md
│   ├── bm25.md               # ← planned
│   ├── embeddings.md         # ← planned
│   └── hybrid_approaches.md  # ← future
├── requirements.txt
└── README.md
````


---

## Current & Planned Text Ranking Approaches

The text search component is being evolved in stages — following the standard progression in modern information retrieval:

| Stage | Method                              | Type       | Key Strength                          | Status     | Documentation                          |
|-------|-------------------------------------|------------|---------------------------------------|------------|----------------------------------------|
| 1     | TF-IDF + Cosine Similarity          | Lexical    | Fast, interpretable keyword baseline  | Active     | [docs/tfidf_cosine_similarity.md]      |
| 2     | BM25 (Okapi BM25)                   | Lexical    | Better length norm + term saturation  | Next       | docs/bm25.md (planned)                 |
| 3     | Dense Text Embeddings + Cosine      | Semantic   | True meaning, synonyms, paraphrasing  | Planned    | docs/embeddings.md (planned)           |
| 4     | Hybrid (BM25 + Embeddings)          | Combined   | Precision + recall balance            | Future     | docs/hybrid_approaches.md (future)     |
| 5+    | Reranking, Sparse Learned (SPLADE), fine-tuning, etc. | Advanced | Highest quality                       | Stretch    | —                                      |

**Current implementation:** Stage 1 — TF-IDF + cosine similarity  
**Next recommended step:** Stage 2 — BM25 (very high impact for minimal effort)

For detailed explanations and comparisons, see the `/docs/` folder.

---

## Key Components

### 1. Data Management (`data_management.py`)

Handles:

- Multi-file CSV ingestion (keyword-based filtering)  
- Automatic column type inference  
- Missing value & uniqueness profiling  
- Column grouping by type  
- Safe PII anonymization  
- Text field combination/preparation for search  

Core function example:

```python
combined_df, summary_df, type_groups = dm.summarize_columns_from_sources(
    job_files, folder="../Data/"
)
```
### 2. Text Search & Ranking (ml_text_search.py)
Current flagship function:
````
results = calc_tfidf_cosine_similarity(
    clean_df,
    corpus=job_text,
    query="remote python data scientist",
    top_n=15,
    extra_columns=["title", "company_name", "location_city", "is_remote"]
)
````

Future functions (planned):

- calc_bm25_similarity(...)
- calc_embedding_cosine_similarity(...)
- calc_hybrid_similarity(...)

## Getting Started
````
# Create & activate virtual environment
python3 -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
````

Then run the example workflow:
````
python main.py
````

## Future Work (Short/Medium Term)
1. Implement BM25 as the next lexical improvement
2. Add detailed BM25 documentation (docs/bm25.md)
3. Introduce sentence-transformer embeddings
4. Compare lexical vs semantic performance on real queries
5. Build hybrid ranking (weighted fusion or reciprocal rank fusion)
6. Add basic evaluation (nDCG, precision@10, recall@K, etc)
7. Explore query expansion, spell correction, skill synonym handling


## Disclaimer
This project is for educational, analytical, and personal experimentation
purposes only. No proprietary scraping logic is included.