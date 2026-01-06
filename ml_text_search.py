import pandas as pd
import numpy as np
from typing import List, Optional, Union, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calc_tfidf_cosine_similarity11(
        corpus: pd.Series,
        query: str,
        top_n: int = 10
) -> Tuple[np.ndarray, pd.DataFrame]:

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(corpus)

    q = vectorizer.transform([query])

    similarities = cosine_similarity(X, q).flatten()
    top_idx = similarities.argsort()[::-1][:top_n]

    all_similarities = similarities.argsort()[::-1]

    top_similarities = pd.DataFrame({
        "index": top_idx,
        "similarity": similarities[top_idx]
    })

    return all_similarities, top_similarities


def calc_tfidf_cosine_similarity(
        clean_df: pd.DataFrame,
        corpus: pd.Series,
        query: str,
        top_n: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(corpus)

    q = vectorizer.transform([query])

    similarities = cosine_similarity(X, q).flatten()
    top_idx = similarities.argsort()[::-1][:top_n]

    all_similarities = clean_df[similarities.argsort()[::-1],['company_name','description','position']]

    top_similarities = pd.DataFrame({
        "index": top_idx,
        "similarity": similarities[top_idx]
    })

    return all_similarities, top_similarities
