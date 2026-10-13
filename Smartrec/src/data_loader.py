"""Load and validate the movie dataset."""
from pathlib import Path

import pandas as pd

from src.config import DEFAULT_DATASET_PATH, REQUIRED_COLUMNS
from src.exceptions import DatasetNotFoundError, DatasetValidationError
from src.preprocessing import CleaningReport, clean_movies


def validate_columns(df: pd.DataFrame) -> None:
    """Raise DatasetValidationError if required columns are missing."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetValidationError(
            "The dataset is missing required column(s): "
            f"{', '.join(missing)}. Expected: {', '.join(REQUIRED_COLUMNS)}."
        )


def load_raw_movies(path: str | Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """Read the CSV file and check its structure."""
    path = Path(path)
    if not path.exists():
        raise DatasetNotFoundError(f"Dataset file not found: {path}")
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DatasetValidationError(f"The dataset could not be parsed: {exc}") from exc
    validate_columns(df)
    return df


def load_movies(
    path: str | Path = DEFAULT_DATASET_PATH,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Load, validate and clean the dataset."""
    return clean_movies(load_raw_movies(path))
