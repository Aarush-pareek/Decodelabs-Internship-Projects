"""The recommendation pipeline:

preferences -> user profile -> vectors -> cosine similarity -> relevance score
-> ranking -> Top-N (+ explanations)
"""
import numpy as np
import pandas as pd

from src import config
from src.exceptions import EmptyPreferencesError, InvalidPreferencesError
from src.explanation import explain_recommendation
from src.feature_engineering import FeatureSpace
from src.models import (
    MatchDetails,
    Recommendation,
    RecommendationResult,
    UserPreferences,
    UserProfile,
)
from src.similarity import blend_with_rating, calculate_similarity


def build_user_profile(
    preferences: UserPreferences,
    movies: pd.DataFrame,
    feedback: dict[int, int] | None = None,
) -> UserProfile:
    """Convert selections (and optional likes) into weighted preferences.

    Cold start: with no feedback the profile is built ONLY from what the user
    selected. Liked titles (rating >= 4) add a boost to their genres and tags.
    """
    genre_weights = {g: float(s) for g, s in preferences.genres.items()}
    interest_weights = {t: 1.0 for t in preferences.interests}
    liked_count = 0

    for movie_id, score in (feedback or {}).items():
        if score < config.LIKE_THRESHOLD or movie_id not in movies.index:
            continue
        row = movies.loc[movie_id]
        for genre in row["genres"]:
            genre_weights[genre] = genre_weights.get(genre, 0.0) + config.FEEDBACK_GENRE_BOOST
        for tag in row["tags"]:
            interest_weights[tag] = interest_weights.get(tag, 0.0) + config.FEEDBACK_INTEREST_BOOST
        liked_count += 1

    text = preferences.free_text.strip()
    if not (genre_weights or interest_weights or text):
        raise EmptyPreferencesError(
            "Please choose at least one genre or interest, or describe what you want."
        )
    return UserProfile(
        genre_weights=genre_weights,
        interest_weights=interest_weights,
        text=text,
        selected_genres=set(preferences.genres),
        selected_interests=set(preferences.interests),
        liked_count=liked_count,
    )


def rank_recommendations(
    scores: np.ndarray, tiebreak_ratings: np.ndarray, eligible: np.ndarray, top_n: int
) -> np.ndarray:
    """Indices of the best eligible items: score desc, then rating desc."""
    candidates = np.flatnonzero(eligible)
    order = np.lexsort((-tiebreak_ratings[candidates], -np.round(scores[candidates], 6)))
    return candidates[order][:top_n]


class RecommendationEngine:
    """Holds the catalogue and its precomputed item vectors."""

    def __init__(self, movies: pd.DataFrame) -> None:
        self.movies = movies.reset_index(drop=True)
        self.features = FeatureSpace(self.movies)
        self._text_terms = np.array(self.features.text_vocabulary)

    @property
    def available_genres(self) -> list[str]:
        return sorted(self.features.genre_vocabulary)

    @property
    def available_interests(self) -> list[str]:
        """Tags used by at least two movies (keeps the picker meaningful)."""
        counts = self.movies["tags"].explode().value_counts()
        return sorted(counts[counts >= 2].index)

    @property
    def available_languages(self) -> list[str]:
        return sorted(self.movies["language"].unique())

    def _validate_against_catalogue(self, preferences: UserPreferences) -> None:
        unknown_genres = [g for g in preferences.genres if g not in self.features.genre_vocabulary]
        if unknown_genres:
            raise InvalidPreferencesError(f"Unknown genre(s): {', '.join(unknown_genres)}.")
        known_tags = self.features.interest_vocabulary
        unknown_tags = [t for t in preferences.interests if t not in known_tags]
        if unknown_tags:
            raise InvalidPreferencesError(f"Unknown interest(s): {', '.join(unknown_tags)}.")

    def generate_recommendations(
        self, preferences: UserPreferences, feedback: dict[int, int] | None = None
    ) -> RecommendationResult:
        """Run the full pipeline and return explained, ranked recommendations."""
        preferences.validate()
        self._validate_against_catalogue(preferences)
        profile = build_user_profile(preferences, self.movies, feedback)
        warnings: list[str] = []

        user_vectors = {
            "genres": self.features.vectorize_genres(profile.genre_weights),
            "interests": self.features.vectorize_interests(profile.interest_weights),
            "text": self.features.vectorize_text(profile.text) if profile.text else np.zeros(
                len(self._text_terms)),
        }
        if profile.text and not user_vectors["text"].any():
            warnings.append("None of the words you typed were found in the catalogue, "
                            "so the description was ignored.")

        item_matrices = {
            "genres": self.features.genre_matrix,
            "interests": self.features.interest_matrix,
            "text": self.features.text_matrix,
        }
        similarity, block_scores, weights = calculate_similarity(user_vectors, item_matrices)
        ratings = self.movies["rating"].to_numpy(dtype=float)
        final_scores = blend_with_rating(similarity, ratings, preferences.mode)

        eligible = self._eligibility_mask(preferences, similarity, feedback)
        top_idx = rank_recommendations(final_scores, ratings, eligible, preferences.top_n)
        if len(top_idx) < preferences.top_n:
            warnings.append(
                f"Only {len(top_idx)} title(s) matched your preferences and filters; "
                "irrelevant titles were not added to fill the list."
            )

        recommendations = [
            self._build_recommendation(rank, idx, profile, user_vectors, block_scores, weights,
                                       similarity, final_scores, preferences.mode)
            for rank, idx in enumerate(top_idx, start=1)
        ]
        return RecommendationResult(recommendations, profile, int(eligible.sum()), warnings)

    def _eligibility_mask(self, preferences, similarity, feedback) -> np.ndarray:
        mask = similarity >= config.MIN_SIMILARITY
        mask &= self.movies["rating"].to_numpy(dtype=float) >= preferences.min_rating
        if preferences.languages:
            wanted = {lang.lower() for lang in preferences.languages}
            mask &= self.movies["language"].str.lower().isin(wanted).to_numpy()
        for movie_id, score in (feedback or {}).items():
            if score <= config.DISLIKE_THRESHOLD and 0 <= movie_id < len(mask):
                mask[movie_id] = False
        return mask

    def _build_recommendation(self, rank, idx, profile, user_vectors, block_scores, weights,
                              similarity, final_scores, mode) -> Recommendation:
        row = self.movies.iloc[idx]
        matched_genres = [g for g in row["genres"] if g in profile.selected_genres]
        matched_interests = [t for t in row["tags"] if t in profile.selected_interests]
        from_feedback = [g for g in row["genres"] if g in profile.genre_weights
                         and g not in profile.selected_genres]
        from_feedback += [t for t in row["tags"] if t in profile.interest_weights
                          and t not in profile.selected_interests]

        keywords: list[str] = []
        if "text" in weights:
            overlap = (user_vectors["text"] > 0) & (self.features.text_matrix[idx] > 0)
            keywords = list(self._text_terms[overlap])

        details = MatchDetails(
            matched_genres=matched_genres,
            matched_interests=matched_interests,
            matched_keywords=keywords,
            matched_from_feedback=from_feedback,
            block_scores={b: float(s[idx]) for b, s in block_scores.items()},
            block_weights=dict(weights),
            preference_similarity=float(similarity[idx]),
            rating_score=float(row["rating"]) / config.MAX_RATING,
            final_score=float(min(final_scores[idx], 1.0)),
        )
        explanation = explain_recommendation(
            details, float(row["rating"]), mode == config.MODE_BALANCED)
        year = None if pd.isna(row["year"]) else int(row["year"])
        return Recommendation(
            rank=rank, movie_id=int(row["movie_id"]), title=row["title"], year=year,
            genres=list(row["genres"]), tags=list(row["tags"]), overview=row["overview"],
            rating=float(row["rating"]), language=row["language"],
            details=details, explanation=explanation,
        )

    def popular_fallback(self, top_n: int = config.DEFAULT_TOP_N) -> pd.DataFrame:
        """Not personalised: highest-rated titles (a cold-start bypass)."""
        return self.movies.sort_values(["rating", "title"], ascending=[False, True]).head(top_n)
