# TF-IDF and Cosine Similarity for Job Text Search

This project uses **TF-IDF vectorization** combined with **cosine similarity**
to enable basic semantic search over job postings.

This document explains the intuition behind these techniques and why they are
well-suited for early-stage job matching systems.

---

## Why Text Search for Job Postings?

Job postings are largely unstructured text:
- Job titles
- Descriptions
- Requirements
- Responsibilities

Traditional keyword matching is brittle.  
TF-IDF provides a lightweight way to capture *importance* of words across many documents,
while cosine similarity measures how similar two pieces of text are in vector space.

---

## TF-IDF (Term Frequency–Inverse Document Frequency)

TF-IDF converts text into numerical vectors by weighting words based on:

- **Term Frequency (TF):**  
  How often a word appears in a document

- **Inverse Document Frequency (IDF):**  
  How rare that word is across all documents

Intuition:
- Common words like *“the”* or *“job”* get low weight
- More informative words like *“python”*, *“sql”*, or *“remote”* get higher weight

This helps distinguish documents based on meaningful content rather than noise.

---

## Vector Representation

After TF-IDF transformation:
- Each job posting becomes a high-dimensional vector
- Each dimension corresponds to a term in the vocabulary
- Values represent TF-IDF weights

These vectors allow us to apply geometric similarity measures.

---

## Cosine Similarity

Cosine similarity measures the **angle** between two vectors rather than their magnitude.

- Value ranges from **0 to 1**
- 1 = identical direction (very similar text)
- 0 = orthogonal (unrelated text)

Why cosine similarity?
- Job descriptions vary greatly in length
- Cosine similarity normalizes for document size
- Focuses on shared informative terms

---

## Applying This to Job Search

In this project:

1. Job descriptions are cleaned and combined into a text corpus
2. TF-IDF is fit on the full job corpus
3. User queries are transformed using the same TF-IDF model
4. Cosine similarity scores rank jobs by relevance

Example:

```python
calc_cosine_similarity(
    corpus=job_text_series,
    query="python data scientist remote",
    top_n=10
)
```

The output returns the most relevant job postings based on textual similarity.

## Limitations

- TF-IDF does not capture semantic meaning (synonyms, context)
- Misses relationships like "ML Engineering" $\approx$ "Machine Learning Scientist"
- Vocabulary is fixed after fitting

These limitations motivate future work using  embeddings and transformer-based models.

## Why This Approach
TF-IDF + cosine similarity offers:
- Interpretability
- Speed
- Minimal dependencies
- A strong baseline for text relevance

It provides a clear, extensible foundation for more advanced NLP techniques.

