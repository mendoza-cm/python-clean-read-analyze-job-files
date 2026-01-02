## TF-IDF vs Text Embeddings

This project currently uses **TF-IDF vectorization combined with cosine similarity** as a baseline approach to text-based job matching.

While both TF-IDF and embeddings represent text as vectors, they differ fundamentally in how meaning is captured.

### TF-IDF (Current Approach)

TF-IDF (Term Frequency–Inverse Document Frequency) represents text using:

- One dimension per vocabulary term
- Weights based on term frequency and global rarity
- Sparse, high-dimensional vectors

**Strengths:**
- Simple and interpretable
- Fast to compute
- Strong lexical baseline
- Effective when exact keyword overlap matters

**Limitations:**
- No semantic understanding
- Cannot capture synonyms or paraphrases
- Vocabulary-dependent

Example:
- “data scientist” and “machine learning engineer” may score low similarity
  despite describing similar roles.

---

### Text Embeddings (Future Direction)

Text embeddings are dense vector representations learned from large corpora using neural models (e.g., transformer architectures).

**Key characteristics:**
- Dense, low-dimensional vectors
- Capture semantic meaning
- Context-aware
- Robust to paraphrasing and synonymy

Example:
- “data scientist” and “machine learning engineer” are likely to be close
  in embedding space even without shared words.

---

### Why Start with TF-IDF?

TF-IDF provides a strong, transparent baseline that allows:

- Clear inspection of model behavior
- Rapid iteration
- Validation of data quality and preprocessing

Embedding-based approaches will be layered on top once the data pipeline and
evaluation criteria are well-established.

## Planned Embedding-Based Similarity Pipeline

A future iteration of this project will replace TF-IDF vectors with
pretrained text embeddings while retaining cosine similarity as the
similarity metric.

Proposed pipeline:

1. Clean and combine job text fields
2. Generate embeddings for job postings and user queries
3. Compute cosine similarity in embedding space
4. Rank job postings by semantic relevance

This design preserves the existing pipeline structure while
improving semantic understanding.

> For implementation details of TF-IDF + cosine similarity used in this project,
> see [TF-IDF Cosine Similarity](tfidf_cosine_similarity.md).
