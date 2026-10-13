"""Confidence filtering and the 80% validation rule.

Two distinct things are computed, on purpose:

* WORD FILTER (brief, slide 16): a word below the threshold is dropped from the
  final text.
* VALIDATION (brief, slide 17): the headline confidence is the mean over ALL
  recognised words, before filtering. Averaging only the survivors would always
  give >= 80% and would be a false pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from src import config
from src.ocr_engine import WordResult


class Status(str, Enum):
    VALIDATED = "validated"
    BELOW_THRESHOLD = "below_threshold"
    NO_TEXT = "no_text"


@dataclass(frozen=True)
class ValidationResult:
    status: Status
    average: float | None  # mean confidence of all words, 0-100; None if no words
    required: float
    total_words: int
    kept_words: int
    dropped_words: int
    min_confidence: float | None
    max_confidence: float | None
    message: str

    @property
    def passed(self) -> bool:
        return self.status is Status.VALIDATED


def split_by_confidence(
    words: list[WordResult], threshold: float = config.CONFIDENCE_THRESHOLD
) -> tuple[list[WordResult], list[WordResult]]:
    """Return (kept, dropped): kept words have confidence >= threshold."""
    kept = [w for w in words if w.confidence >= threshold]
    dropped = [w for w in words if w.confidence < threshold]
    return kept, dropped


def average_confidence(words: list[WordResult]) -> float | None:
    """Mean confidence of the given words rounded to 2 decimals, or None if empty."""
    if not words:
        return None
    return round(sum(w.confidence for w in words) / len(words), 2)


def format_confidence(value: float | None) -> str:
    """Show one decimal, truncating rather than rounding up, so 79.96 is never
    displayed as '80.0%' next to a BELOW THRESHOLD result."""
    if value is None:
        return "n/a"
    return f"{math.floor(value * 10) / 10:.1f}%"


def validate(
    words: list[WordResult], threshold: float = config.CONFIDENCE_THRESHOLD
) -> ValidationResult:
    """Apply the validation rule to the full list of recognised words."""
    kept, dropped = split_by_confidence(words, threshold)

    if not words:
        return ValidationResult(
            Status.NO_TEXT, None, threshold, 0, 0, 0, None, None,
            "No text was recognised, so there is no confidence to validate.",
        )

    average = average_confidence(words)
    lowest = min(w.confidence for w in words)
    highest = max(w.confidence for w in words)

    if average >= threshold:
        status = Status.VALIDATED
        message = (f"Average confidence {format_confidence(average)} meets the "
                   f"required minimum of {threshold:.0f}%.")
    else:
        status = Status.BELOW_THRESHOLD
        message = (f"Average confidence {format_confidence(average)} is below the "
                   f"required minimum of {threshold:.0f}%. Review the output manually "
                   f"or try another page mode or preprocessing setting.")

    return ValidationResult(
        status, average, threshold, len(words), len(kept), len(dropped),
        lowest, highest, message,
    )
