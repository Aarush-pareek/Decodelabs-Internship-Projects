"""Clean the raw movie table and report everything that was changed."""
from dataclasses import dataclass, field

import pandas as pd

from src.config import GENRE_ALIASES, LIST_SEPARATOR, MAX_RATING
from src.exceptions import DatasetValidationError


@dataclass
class CleaningReport:
    """Counts and messages describing what cleaning did (nothing is silent)."""

    rows_loaded: int = 0
    rows_kept: int = 0
    dropped_missing_title: int = 0
    dropped_missing_genres: int = 0
    dropped_invalid_rating: int = 0
    dropped_duplicates: int = 0
    missing_overviews: int = 0
    missing_years: int = 0
    messages: list[str] = field(default_factory=list)


def normalize_genre(name: str) -> str:
    """Map spelling variants to one canonical genre name."""
    cleaned = name.strip()
    return GENRE_ALIASES.get(cleaned.lower(), cleaned.title())


def _split_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(LIST_SEPARATOR) if part.strip()]


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def clean_movies(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Return a clean table (list columns, movie_id) and a CleaningReport."""
    report = CleaningReport(rows_loaded=len(raw))
    df = raw.copy()

    for col in ("title", "genres", "overview", "language", "tags"):
        df[col] = df[col].fillna("").astype(str).str.strip()

    missing_title = df["title"] == ""
    report.dropped_missing_title = int(missing_title.sum())
    df = df[~missing_title].copy()

    df["genres"] = df["genres"].map(
        lambda s: _unique([normalize_genre(g) for g in _split_list(s)])
    )
    no_genres = df["genres"].map(len) == 0
    report.dropped_missing_genres = int(no_genres.sum())
    df = df[~no_genres].copy()

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    bad_rating = df["rating"].isna() | (df["rating"] < 0) | (df["rating"] > MAX_RATING)
    report.dropped_invalid_rating = int(bad_rating.sum())
    df = df[~bad_rating].copy()

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    report.missing_years = int(df["year"].isna().sum())
    report.missing_overviews = int((df["overview"] == "").sum())

    df["tags"] = df["tags"].map(lambda s: _unique([t.lower() for t in _split_list(s)]))
    df["language"] = df["language"].replace("", "Unknown").str.title()

    key = df["title"].str.lower() + "|" + df["year"].astype(str)
    duplicated = key.duplicated(keep="first")
    report.dropped_duplicates = int(duplicated.sum())
    df = df[~duplicated]

    df = df.reset_index(drop=True)
    df["movie_id"] = df.index
    report.rows_kept = len(df)

    if df.empty:
        raise DatasetValidationError("No valid movie rows remain after cleaning.")

    _add_messages(report)
    return df, report


def _add_messages(report: CleaningReport) -> None:
    checks = [
        (report.dropped_missing_title, "row(s) dropped: missing title"),
        (report.dropped_missing_genres, "row(s) dropped: no genres"),
        (report.dropped_invalid_rating, "row(s) dropped: rating missing or outside 0-10"),
        (report.dropped_duplicates, "duplicate row(s) removed (same title and year)"),
        (report.missing_overviews, "movie(s) have no overview (genres/tags still used)"),
        (report.missing_years, "movie(s) have no valid year"),
    ]
    report.messages = [f"{n} {text}" for n, text in checks if n]
