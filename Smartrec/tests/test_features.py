"""TF-IDF feature space."""
import numpy as np

from src.feature_engineering import FeatureSpace


def test_matrix_shapes_match_catalogue(small_movies):
    space = FeatureSpace(small_movies)
    n = len(small_movies)
    assert space.genre_matrix.shape == (n, len(space.genre_vocabulary))
    assert space.interest_matrix.shape == (n, len(space.interest_vocabulary))
    assert space.text_matrix.shape == (n, len(space.text_vocabulary))


def test_user_and_item_share_one_vocabulary(small_movies):
    space = FeatureSpace(small_movies)
    vector = space.vectorize_genres({"Sci-Fi": 3.0})
    assert vector.shape[0] == space.genre_matrix.shape[1]
    assert np.count_nonzero(vector) == 1


def test_rare_genre_gets_higher_weight_than_common_genre(small_movies):
    space = FeatureSpace(small_movies)
    vocab = space.genre_vocabulary
    vector = space.vectorize_genres({"Sci-Fi": 1.0, "Comedy": 1.0})
    # Sci-Fi appears in 2 movies, Comedy in 1: the rarer term must weigh more.
    assert vector[vocab.index("Comedy")] > vector[vocab.index("Sci-Fi")]


def test_unknown_terms_give_zero_vector(small_movies):
    space = FeatureSpace(small_movies)
    assert not space.vectorize_genres({"Western": 3.0}).any()
    assert not space.vectorize_text("zzzz qqqq").any()


def test_strength_scales_weight(small_movies):
    space = FeatureSpace(small_movies)
    low = space.vectorize_genres({"Sci-Fi": 1.0}).max()
    high = space.vectorize_genres({"Sci-Fi": 4.0}).max()
    assert np.isclose(high, 4 * low)
