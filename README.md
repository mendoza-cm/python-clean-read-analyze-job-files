# Python Job File Analysis & Text Search

This project focuses on cleaning, profiling, and analyzing job posting data scraped from multiple employment websites.  
It demonstrates a full pipeline from raw CSV ingestion through exploratory analysis and basic NLP-based text search.

> **Note:** Data was collected via custom web scraping pipelines (not included).  
> Only the cleaned CSV outputs and downstream analysis code are part of this repository.

---

## Project Goals

- Clean and standardize heterogeneous job posting data  
- Infer column types and summarize missingness  
- Prepare free-text fields for machine learning workflows  
- Implement a basic TF-IDF + cosine similarity job search engine  
- Serve as a foundation for future NLP / ML job-matching models  

---

## Repository Structure

```
python-clean-read-analyze-job-files/
├── dataManagement.py # Data ingestion, cleaning, profiling utilities
├── ml_text_search.py # TF-IDF + cosine similarity text search
├── main.py # Example analysis workflow
├── Data/ # Scraped CSV job files (raw inputs)
├── Results/ # Generated outputs (plots, summaries)
├── requirements.txt
├── README.md
```

---

## Key Components

### 1. Data Management (`dataManagement.py`)

Includes utilities for:

- Loading multiple CSVs by filename keywords  
- Inferring column types (numeric, categorical, text, boolean, datetime)  
- Summarizing missingness and uniqueness  
- Grouping columns by inferred type  
- Cleaning and combining text fields for NLP workflows  

Example usage:

```python
combined_df, summary_df, type_groups = summarize_columns_from_sources(
    job_files,
    folder="../Data/"
)
```
The resulting text representations are suitable for TF-IDF, embeddings, or downstream machine learning models.

### 2. Text-Based Job Search (`ml_text_search.py`)

Implements a simple semantic search using:

- TF-IDF vectorization
- Cosine similarity scoring
- Users can enter free-text queries and retrieve the most relevant job postings.

Example:
```
calc_cosine_similarity(
    corpus=job_text_series,
    query="python data scientist remote",
    top_n=10
)
```

## Future Work
- Sentence-transormer embeddings for semantic search
- supervised job-classification
- Clustering similar roles
- Feature Engineering for compensation modeling
- Web-based UI for interactive search

## Environment Work
````
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
````


## Disclaimer
This project is for analytical and educational purposes only. 
Scraping logic and proprietary pipelines are intentionally excluded.