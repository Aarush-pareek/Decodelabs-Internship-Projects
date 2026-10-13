"""Plain data containers shared across the recommendation pipeline."""
from dataclasses import dataclass, field

from src import config
from src.exceptions import InvalidPreferencesError


@dataclass
class UserPreferences:
    """What the user told us. Genre values are strengths (1-5)."""

    genres: dict[str, float] = field(default_factory=dict)
    interests: list[str] = field(default_factory=list)
    free_text: str = ""
    min_rating: float = 0.0
    languages: list[str] = field(default_factory=list)  # empty = any language
    top_n: int = config.DEFAULT_TOP_N
    mode: str = config.MODE_BEST_MATCH

    def has_signal(self) -> bool:
        """True if at least one preference block is filled in."""
        return bool(self.genres or self.interests or self.free_text.strip())

    def validate(self) -> None:
        """Raise InvalidPreferencesError for malformed values."""
        if not 1 <= int(self.top_n) <= config.MAX_TOP_N:
            raise InvalidPreferencesError(
                f"Number of recommendations must be between 1 and {config.MAX_TOP_N}."
            )
        if not 0 <= float(self.min_rating) <= config.MAX_RATING:
            raise InvalidPreferencesError(
                f"Minimum rating must be between 0 and {config.MAX_RATING:g}."
            )
        if self.mode not in config.MODES:
            raise InvalidPreferencesError(f"Unknown recommendation mode: {self.mode!r}.")
        for genre, strength in self.genres.items():
            if not config.MIN_STRENGTH <= float(strength) <= config.MAX_STRENGTH:
                raise InvalidPreferencesError(
                    f"Strength for '{genre}' must be between "
                    f"{config.MIN_STRENGTH:g} and {config.MAX_STRENGTH:g}."
                )


@dataclass
class UserProfile:
    """Numeric-ready representation of the user (before vectorisation)."""

    genre_weights: dict[str, float]
    interest_weights: dict[str, float]
    text: str
    selected_genres: set[str]
    selected_interests: set[str]
    liked_count: int = 0

    @property
    def is_cold_start(self) -> bool:
        """True when the profile comes only from selected preferences."""
        return self.liked_count == 0


@dataclass
class MatchDetails:
    """Everything the algorithm knows about why an item scored as it did."""

    matched_genres: list[str]
    matched_interests: list[str]
    matched_keywords: list[str]
    matched_from_feedback: list[str]
    block_scores: dict[str, float]
    block_weights: dict[str, float]
    preference_similarity: float
    rating_score: float
    final_score: float

    @property
    def contributions(self) -> dict[str, float]:
        """Weighted share of each block in the preference similarity."""
        return {b: self.block_weights[b] * s for b, s in self.block_scores.items()}

    @property
    def matched_count(self) -> int:
        return (
            len(self.matched_genres)
            + len(self.matched_interests)
            + len(self.matched_keywords)
        )


@dataclass
class Explanation:
    """Human-readable explanation generated from MatchDetails."""

    label: str
    matched_features: list[str]
    reason: str


@dataclass
class Recommendation:
    """One ranked recommendation."""

    rank: int
    movie_id: int
    title: str
    year: int | None
    genres: list[str]
    tags: list[str]
    overview: str
    rating: float
    language: str
    details: MatchDetails
    explanation: Explanation

    @property
    def match_percent(self) -> int:
        return int(round(self.details.final_score * 100))


@dataclass
class RecommendationResult:
    """Ranked list plus context the UI needs."""

    recommendations: list[Recommendation]
    profile: UserProfile
    candidates_considered: int
    warnings: list[str] = field(default_factory=list)
