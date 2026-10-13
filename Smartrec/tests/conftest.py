"""Shared fixtures: a tiny hand-made catalogue and the real dataset."""
import pandas as pd
import pytest

from src.data_loader import load_movies
from src.preprocessing import clean_movies
from src.recommendation_engine import RecommendationEngine


@pytest.fixture
def raw_movies() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "title": ["Star Quest", "Love Letters", "Laugh Riot", "Dark Orbit", "Heist Night"],
            "year": [2010, 2012, 2015, 2018, 2020],
            "genres": ["Sci-Fi|Adventure", "Romance|Drama", "Comedy", "Sci-Fi|Thriller",
                       "Crime|Thriller"],
            "overview": [
                "A crew explores a distant galaxy.",
                "Two writers fall in love by mail.",
                "Friends cause chaos at a wedding.",
                "A station orbiting a dying star goes silent.",
                "A crew plans a perfect robbery.",
            ],
            "rating": [8.0, 7.0, 6.5, 7.5, 8.2],
            "language": ["English", "French", "English", "English", "English"],
            "tags": ["space|crew", "love story|letters", "wedding|friendship", "space|survival",
                     "heist|crew"],
        }
    )


@pytest.fixture
def small_movies(raw_movies):
    movies, _ = clean_movies(raw_movies)
    return movies


@pytest.fixture
def small_engine(small_movies) -> RecommendationEngine:
    return RecommendationEngine(small_movies)


@pytest.fixture
def real_engine() -> RecommendationEngine:
    movies, _ = load_movies()
    return RecommendationEngine(movies)
