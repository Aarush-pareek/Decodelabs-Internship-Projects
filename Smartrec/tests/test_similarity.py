"""Cosine similarity and score blending."""
import numpy as np
import pytest

from src import config
from src.exceptions import SimilarityError
from src.similarity import blend_with_rating, calculate_similarity, cosine_similarity_scores


def test_identical_vectors_score_one():
    user = np.array([1.0, 2.0, 0.0])
    assert np.isclose(cosine_similarity_scores(user, np.array([[1.0, 2.0, 0.0]]))[0], 1.0)


def test_orthogonal_vectors_score_zero():
    user = np.array([1.0, 0.0])
    assert cosine_similarity_scores(user, np.array([[0.0, 5.0]]))[0] == 0.0


def test_known_value_matches_hand_calculation():
    # cos = (1*1 + 1*0) / (sqrt(2) * 1) = 0.7071...
    score = cosine_similarity_scores(np.array([1.0, 1.0]), np.array([[1.0, 0.0]]))[0]
    assert np.isclose(score, 1 / np.sqrt(2))


def test_scale_invariance():
    user = np.array([1.0, 2.0])
    scores = cosine_similarity_scores(user, np.array([[1.0, 2.0], [10.0, 20.0]]))
    assert np.isclose(scores[0], scores[1])


def test_zero_user_vector_returns_zeros_without_error():
    scores = cosine_similarity_scores(np.zeros(2), np.array([[1.0, 1.0]]))
    assert scores.tolist() == [0.0]


def test_size_mismatch_raises_similarity_error():
    with pytest.raises(SimilarityError):
        cosine_similarity_scores(np.array([1.0, 2.0, 3.0]), np.array([[1.0, 2.0]]))


def test_only_active_blocks_are_blended_and_weights_sum_to_one():
    items = {"genres": np.eye(2), "interests": np.eye(2)}
    users = {"genres": np.array([1.0, 0.0]), "interests": np.zeros(2)}
    similarity, block_scores, weights = calculate_similarity(users, items)
    assert list(weights) == ["genres"]
    assert np.isclose(sum(weights.values()), 1.0)
    assert similarity.tolist() == [1.0, 0.0]
    assert "interests" not in block_scores


def test_no_active_blocks_gives_zero_similarity():
    items = {"genres": np.eye(2)}
    similarity, _, weights = calculate_similarity({"genres": np.zeros(2)}, items)
    assert similarity.tolist() == [0.0, 0.0] and weights == {}


def test_balanced_mode_adds_rating_component():
    sim = np.array([0.5])
    best = blend_with_rating(sim, np.array([10.0]), config.MODE_BEST_MATCH)
    balanced = blend_with_rating(sim, np.array([10.0]), config.MODE_BALANCED)
    assert best[0] == 0.5
    assert np.isclose(balanced[0], 0.8 * 0.5 + 0.2 * 1.0)
