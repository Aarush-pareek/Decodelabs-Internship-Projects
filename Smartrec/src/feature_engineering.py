"""Turn items and user preferences into vectors in one shared TF-IDF space.

Three feature blocks are built from the movie table:
  * genres    - one dimension per genre
  * interests - one dimension per tag (e.g. "heist", "space")
  * text      - words from overview + tags + genres (stop words removed)

Each block is a TF-IDF matrix (rows = movies). A user is mapped into the SAME
vocabulary, so cosine similarity compares like with like.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def _identity(tokens: list[str]) -> list[str]:
    """Analyzer for columns that already contain lists of tokens."""
    return tokens


def _make_list_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer=_identity)


class FeatureSpace:
    """Fitted TF-IDF vocabularies and item matrices for the catalogue."""

    def __init__(self, movies: pd.DataFrame) -> None:
        self._genre_vec = _make_list_vectorizer()
        self.genre_matrix = self._genre_vec.fit_transform(movies["genres"]).toarray()

        self._interest_vec = _make_list_vectorizer()
        self.interest_matrix = self._interest_vec.fit_transform(movies["tags"]).toarray()

        documents = (
            movies["overview"] + " " + movies["tags"].map(" ".join) + " "
            + movies["genres"].map(" ".join)
        ).str.lower()
        self._text_vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        self.text_matrix = self._text_vec.fit_transform(documents).toarray()

    @property
    def genre_vocabulary(self) -> list[str]:
        return list(self._genre_vec.get_feature_names_out())

    @property
    def interest_vocabulary(self) -> list[str]:
        return list(self._interest_vec.get_feature_names_out())

    @property
    def text_vocabulary(self) -> list[str]:
        return list(self._text_vec.get_feature_names_out())

    def vectorize_genres(self, weights: dict[str, float]) -> np.ndarray:
        """User genre strengths -> vector (strength x IDF per genre)."""
        return self._weighted_vector(self._genre_vec, weights)

    def vectorize_interests(self, weights: dict[str, float]) -> np.ndarray:
        """User interest weights -> vector (weight x IDF per tag)."""
        return self._weighted_vector(self._interest_vec, weights)

    def vectorize_text(self, text: str) -> np.ndarray:
        """Free text -> TF-IDF vector in the item text vocabulary."""
        return self._text_vec.transform([text.lower()]).toarray()[0]

    @staticmethod
    def _weighted_vector(vectorizer: TfidfVectorizer, weights: dict[str, float]) -> np.ndarray:
        vector = np.zeros(len(vectorizer.vocabulary_), dtype=float)
        for term, weight in weights.items():
            index = vectorizer.vocabulary_.get(term)
            if index is not None:
                vector[index] = weight * vectorizer.idf_[index]
        return vector
