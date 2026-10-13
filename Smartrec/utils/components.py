"""HTML builders for the dashboard (all dynamic text is escaped)."""
from src.models import Recommendation
from utils.helpers import chips_html, esc

STEPS = [
    ("1", "Choose preferences"),
    ("2", "Convert to a vector"),
    ("3", "Cosine similarity"),
    ("4", "Rank items"),
    ("5", "Top-N + reasons"),
]


def hero_html() -> str:
    return (
        '<div class="hero"><h1>Discover content that matches your interests.</h1>'
        "<p>SmartRec analyzes your preferences and ranks movies by similarity, "
        "then shows exactly why each one was recommended.</p></div>"
    )


def steps_html() -> str:
    return '<div class="steps">' + "".join(
        f'<div class="step"><b>STEP {n}</b><div>{esc(text)}</div></div>' for n, text in STEPS
    ) + "</div>"


def card_html(rec: Recommendation, liked: bool = False) -> str:
    """One recommendation card."""
    year = f" ({rec.year})" if rec.year else ""
    badge = '<span class="liked">Liked</span>' if liked else ""
    matched = chips_html(rec.explanation.matched_features, "chip chip-match") or (
        '<span class="muted">Related through similar features</span>'
    )
    return (
        '<div class="card">'
        '<div class="card-top">'
        f'<div class="rank">{rec.rank}</div>'
        '<div class="card-title">'
        f'<div class="title">{esc(rec.title)}<span class="year">{esc(year)}</span>{badge}</div>'
        f'<div>{chips_html(rec.genres)}</div></div>'
        f'<div class="match"><div class="match-value">{rec.match_percent}%</div>'
        f'<div class="match-label">{esc(rec.explanation.label)}</div></div>'
        '</div>'
        f'<div class="bar"><div class="bar-fill" style="width:{rec.match_percent}%"></div></div>'
        f'<div class="meta-line">Rating {rec.rating:.1f} / 10 &middot; {esc(rec.language)}</div>'
        f'<p class="overview">{esc(rec.overview)}</p>'
        '<div class="matched-label">Matching your preferences</div>'
        f'<div>{matched}</div>'
        f'<div class="why"><strong>Why recommended</strong><br>{esc(rec.explanation.reason)}</div>'
        '</div>'
    )
