import pandas as pd
import numpy as np
from typing import List, Optional, Union, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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