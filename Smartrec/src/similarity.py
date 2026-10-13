"""Cosine similarity and score blending, implemented from scratch with NumPy."""
import numpy as np

from src import config
from src.exceptions import SimilarityError


def cosine_similarity_scores(user_vector: np.ndarray, item_matrix: np.ndarray) -> np.ndarray:
    """cos(theta) = (A . B) / (|A| |B|) between the user and every item row.

    Zero-length vectors give a score of 0 instead of a division error.
    """
    if item_matrix.ndim != 2 or user_vector.ndim != 1:
        raise SimilarityError("Expected a 1-D user vector and a 2-D item matrix.")
    if item_matrix.shape[1] != user_vector.shape[0]:
        raise SimilarityError(
            f"Vector size mismatch: user has {user_vector.shape[0]} dimensions, "
            f"items have {item_matrix.shape[1]}."
        )
    user_norm = np.linalg.norm(user_vector)
    item_norms = np.linalg.norm(item_matrix, axis=1)
    dots = item_matrix @ user_vector
    denominator = item_norms * user_norm
    scores = np.divide(dots, denominator, out=np.zeros_like(dots, dtype=float),
                       where=denominator > 0)
    return np.clip(scores, 0.0, 1.0)


def calculate_similarity(
    user_vectors: dict[str, np.ndarray],
    item_matrices: dict[str, np.ndarray],
    base_weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    """Blend per-block cosine scores into one preference similarity.

    Only blocks where the user gave a signal (non-zero vector) take part, and
    their weights are re-normalised to sum to 1.

    Returns (preference_similarity, block_scores, effective_weights).
    """
    base_weights = base_weights or config.BLOCK_WEIGHTS
    active = [b for b, v in user_vectors.items() if np.linalg.norm(v) > 0]
    n_items = next(iter(item_matrices.values())).shape[0]
    if not active:
        return np.zeros(n_items), {}, {}

    total = sum(base_weights[b] for b in active)
    weights = {b: base_weights[b] / total for b in active}
    block_scores = {b: cosine_similarity_scores(user_vectors[b], item_matrices[b]) for b in active}
    similarity = sum(weights[b] * block_scores[b] for b in active)
    return np.clip(similarity, 0.0, 1.0), block_scores, weights


def blend_with_rating(similarity: np.ndarray, ratings: np.ndarray, mode: str) -> np.ndarray:
    """Best match: similarity only. Balanced: 0.8 x similarity + 0.2 x rating/10."""
    if mode == config.MODE_BALANCED:
        rating_score = ratings / config.MAX_RATING
        return (
            config.BALANCED_SIMILARITY_WEIGHT * similarity
            + config.BALANCED_RATING_WEIGHT * rating_score
        )
    return similarity
