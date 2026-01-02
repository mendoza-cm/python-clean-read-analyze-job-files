import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calc_cosine_similarity(
        corpus: pd.Series,
        query: str,
        top_n: int = 10
) -> pd.DataFrame:

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(corpus)

    q = vectorizer.tranform([query])

    similarities = cosine_similarity(X, q).flatten()
    top_idx = similarities.argsort()[::-1][:top_n]

    return pd.DataFrame({
        "index":top_idx,
        "similarity": similarities[top_idx],
    })