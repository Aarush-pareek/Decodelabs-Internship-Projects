"""Generate explanations strictly from the features that produced the score."""
from src.models import Explanation, MatchDetails


def join_natural(items: list[str]) -> str:
    """['a', 'b', 'c'] -> 'a, b and c'."""
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def strength_label(score: float) -> str:
    """Wording for a 0-1 final score."""
    if score >= 0.75:
        return "Strong match"
    if score >= 0.50:
        return "Good match"
    if score >= 0.25:
        return "Moderate match"
    return "Partial match"


def explain_recommendation(details: MatchDetails, rating: float, balanced: bool) -> Explanation:
    """Build the 'why recommended' text from the real matched features."""
    parts = []
    if details.matched_genres:
        parts.append(f"genres {join_natural(details.matched_genres)}")
    if details.matched_interests:
        parts.append(f"interests {join_natural(details.matched_interests)}")
    if details.matched_keywords:
        parts.append(f"keywords {join_natural(details.matched_keywords)}")

    sentences = []
    if parts:
        sentences.append(f"It matches your {join_natural(parts)}.")
    if details.matched_from_feedback:
        sentences.append(
            f"It also shares {join_natural(details.matched_from_feedback)} "
            "with titles you liked."
        )
    if not sentences:
        sentences.append("It is related to your preferences through similar features.")
    if balanced:
        sentences.append(f"Its {rating:.1f} rating also raised its rank.")

    features = (
        details.matched_genres + details.matched_interests + details.matched_keywords
    )
    return Explanation(
        label=strength_label(details.final_score),
        matched_features=features,
        reason=" ".join(sentences),
    )
