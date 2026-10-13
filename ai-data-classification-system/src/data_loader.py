"""Stage 1 - Dataset loading, validation and profiling.

The loader never assumes a dataset is perfect: every entry point validates the
data and raises :class:`DataValidationError` with a human-readable message that
the UI can display directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import IO

import pandas as pd
from sklearn.datasets import load_iris

from src import config


class DataValidationError(ValueError):
    """Raised when a dataset cannot be used for classification."""


@dataclass(frozen=True)
class Dataset:
    """A tabular dataset plus the metadata the pipeline needs."""

    name: str
    frame: pd.DataFrame
    target_column: str

    @property
    def feature_columns(self) -> list[str]:
        """Numeric columns that can be used as model inputs."""
        return [
            column
            for column in self.frame.columns
            if column != self.target_column and pd.api.types.is_numeric_dtype(self.frame[column])
        ]


def load_builtin_iris() -> Dataset:
    """Load the Iris benchmark bundled with scikit-learn."""
    bunch = load_iris(as_frame=True)
    frame = bunch.frame.copy()
    # Use short, readable column names instead of "sepal length (cm)".
    frame.columns = [name.replace(" (cm)", "").replace(" ", "_") for name in frame.columns]
    frame["species"] = frame["target"].map(dict(enumerate(bunch.target_names)))
    frame = frame.drop(columns="target")
    return Dataset(name=config.DATASET_IRIS_BUILTIN, frame=frame, target_column="species")


def read_csv_frame(source: str | Path | IO[bytes]) -> pd.DataFrame:
    """Read a CSV into a DataFrame with tidy column names.

    Raises:
        DataValidationError: If the file is missing, empty or not a valid CSV.
    """
    if isinstance(source, (str, Path)) and not Path(source).is_file():
        raise DataValidationError(f"Dataset file not found: {Path(source).name}")

    try:
        frame = pd.read_csv(source)
    except pd.errors.EmptyDataError as error:
        raise DataValidationError("The CSV file is empty.") from error
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise DataValidationError(f"The file is not a valid CSV: {error}") from error

    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def load_csv(
    source: str | Path | IO[bytes],
    target_column: str | None = None,
    name: str | None = None,
) -> Dataset:
    """Load a CSV file (path or uploaded file object) and validate it.

    Args:
        source: File path or file-like object (e.g. a Streamlit upload).
        target_column: Column holding the class labels. Defaults to the last column.
        name: Display name for the dataset.

    Raises:
        DataValidationError: If the file is missing, unreadable, empty or unsuitable.
    """
    frame = read_csv_frame(source)
    target = target_column if target_column is not None else frame.columns[-1]
    dataset_name = name or (Path(source).name if isinstance(source, (str, Path)) else "Uploaded CSV")
    dataset = Dataset(name=dataset_name, frame=frame, target_column=target)
    validate_dataset(dataset)
    return dataset


def validate_dataset(dataset: Dataset) -> None:
    """Check that the dataset is structurally suitable for classification.

    Raises:
        DataValidationError: With a message explaining what is wrong.
    """
    frame, target = dataset.frame, dataset.target_column

    if frame.empty:
        raise DataValidationError("The dataset contains no rows.")
    if target not in frame.columns:
        raise DataValidationError(
            f"Target column '{target}' was not found. Available columns: {', '.join(frame.columns)}"
        )
    if len(dataset.feature_columns) == 0:
        raise DataValidationError("No numeric feature columns were found. This project supports numeric features.")

    labelled = frame[target].dropna()
    if len(labelled) < config.MIN_ROWS:
        raise DataValidationError(
            f"At least {config.MIN_ROWS} labelled rows are required; found {len(labelled)}."
        )

    class_counts = labelled.value_counts()
    if len(class_counts) < 2:
        raise DataValidationError("The target column needs at least two different classes.")
    if len(class_counts) > config.MAX_CLASSES:
        raise DataValidationError(
            f"Column '{target}' has {len(class_counts)} distinct values; at most {config.MAX_CLASSES} classes are "
            "supported. It may be a continuous or ID column rather than a class label."
        )
    if class_counts.min() < config.MIN_ROWS_PER_CLASS:
        rare = class_counts.idxmin()
        raise DataValidationError(
            f"Class '{rare}' has fewer than {config.MIN_ROWS_PER_CLASS} samples, so it cannot be split reliably."
        )


def summarize_dataset(dataset: Dataset) -> dict[str, object]:
    """Compute the headline statistics shown in the dashboard."""
    frame = dataset.frame
    return {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "features": len(dataset.feature_columns),
        "classes": int(frame[dataset.target_column].nunique()),
        "missing_total": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
    }


def data_quality_table(dataset: Dataset) -> pd.DataFrame:
    """One row per column: data type, missing values and unique values."""
    frame = dataset.frame
    return pd.DataFrame(
        {
            "Column": frame.columns,
            "Role": ["Target" if column == dataset.target_column else "Feature" for column in frame.columns],
            "Data type": [str(dtype) for dtype in frame.dtypes],
            "Missing": frame.isna().sum().to_numpy(),
            "Missing %": (frame.isna().mean() * 100).round(2).to_numpy(),
            "Unique values": frame.nunique().to_numpy(),
        }
    )


def class_distribution(dataset: Dataset) -> pd.Series:
    """Number of samples per class, ordered by class name."""
    return dataset.frame[dataset.target_column].value_counts().sort_index()
