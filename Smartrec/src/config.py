"""Central configuration for SmartRec (paths, weights and thresholds)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "movies.csv"

REQUIRED_COLUMNS = ("title", "year", "genres", "overview", "rating", "language", "tags")
LIST_SEPARATOR = "|"
MAX_RATING = 10.0

# How much each preference block contributes to the preference similarity.
# Only blocks the user actually filled in take part; weights are re-normalised.
BLOCK_WEIGHTS = {"genres": 0.5, "interests": 0.3, "text": 0.2}

# Recommendation modes.
MODE_BEST_MATCH = "Best match"
MODE_BALANCED = "Balanced"
MODES = (MODE_BEST_MATCH, MODE_BALANCED)
BALANCED_SIMILARITY_WEIGHT = 0.8
BALANCED_RATING_WEIGHT = 0.2

# Optional feedback (session-only, explicit ratings).
LIKE_SCORE = 5
DISLIKE_SCORE = 1
LIKE_THRESHOLD = 4
DISLIKE_THRESHOLD = 2
FEEDBACK_GENRE_BOOST = 1.0
FEEDBACK_INTEREST_BOOST = 1.0

# Items below this similarity are treated as "no meaningful match".
MIN_SIMILARITY = 0.01
DEFAULT_TOP_N = 5
MAX_TOP_N = 20
MIN_STRENGTH = 1.0
MAX_STRENGTH = 5.0
DEFAULT_STRENGTH = 3.0

GENRE_ALIASES = {
    "sci-fi": "Sci-Fi",
    "sci fi": "Sci-Fi",
    "scifi": "Sci-Fi",
    "science fiction": "Sci-Fi",
    "romcom": "Romance",
    "animated": "Animation",
    "sports": "Sport",
    "musical": "Music",
}
