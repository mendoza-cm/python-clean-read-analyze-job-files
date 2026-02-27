import pandas as pd
import numpy as np
from typing import List, Optional, Union, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from data_management import normalize_scores


def calc_tfidf_cosine_similarity(
        clean_df: pd.DataFrame,
        corpus: pd.Series,
        query: str,
        top_n: int = 10,
        extra_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Returns a DataFrame with the top N most similar jobs, including:
    - original_index (to map back to clean_df if needed)
    - similarity score
    - any extra columns requested via extra_columns

    Returns empty DataFrame if no results or error.
    """
    if extra_columns is None:
        extra_columns = []

    # Safety: make sure requested columns exist
    available_extra = [col for col in extra_columns if col in clean_df.columns]
    if len(available_extra) < len(extra_columns):
        missing = set(extra_columns) - set(available_extra)
        print(f"Warning: Requested columns not found: {missing}")

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(corpus.fillna(''))
    q = vectorizer.transform([query])

    similarities = cosine_similarity(X, q).flatten()

    # Get indices sorted by descending similarity
    sorted_idx = similarities.argsort()[::-1]
    top_indices = sorted_idx[:top_n]

    # Build result DataFrame
    result = (
        clean_df.iloc[top_indices]
        .reset_index(names='original_index')  # preserve original index
        .assign(similarity=similarities[top_indices])
    )

    # Select only the columns we want (in desired order)
    columns_to_keep = ['original_index', 'similarity'] + available_extra
    result = result[columns_to_keep]

    return result


def calc_semantic_similarity(corpus, query, top_n=10):
    """
    Semantic similarity using TF-IDF + cosine similarity.

    Returns:
        list of (index, score)
    """

    vectorizer = TfidfVectorizer(stop_words="english")

    X = vectorizer.fit_transform(corpus.fillna(""))
    q = vectorizer.transform([query])

    similarities = cosine_similarity(X, q).flatten()

    top_indices = np.argsort(similarities)[::-1][:top_n]

    return [(int(i), similarities[i]) for i in top_indices]


def calc_bm25_similarity(corpus, query, top_n=10):
    """
    Returns: list of (index, score)
    """

    # Tokenize corpus
    tokenized_corpus = [
        doc.split() if isinstance(doc, str) else []
        for doc in corpus.fillna('')
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.split()

    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_n]

    return [(int(i), scores[i]) for i in top_indices]

def calc_hybrid_similarity(corpus, query, top_n=10, semantic_weight=0.7):
    """
    Hybrid: semantic + BM25.
    - semantic_weight: float [0-1], higher favors meaning over keywords.

    Returns: top_n (index, combined_score) tuples.
    """

    bm25_results = calc_bm25_similarity(corpus, query, top_n=len(corpus)) # Full scores

    sem_results = calc_semantic_similarity(corpus, query, top_n=len(corpus))

    bm25_scores = np.zeros(len(corpus))

    sem_scores = np.zeros(len(corpus))

    for i, score in bm25_results: bm25_scores[i] = score

    for i, score in sem_results: sem_scores[i] = score

    bm25_norm = normalize_scores(bm25_scores)

    sem_norm = normalize_scores(sem_scores)

    combined = semantic_weight * sem_norm + (1 - semantic_weight) * bm25_norm

    top_indices = np.argsort(combined)[::-1][:top_n]

    top_results = [(int(i), combined[i]) for i in top_indices]

    return top_results