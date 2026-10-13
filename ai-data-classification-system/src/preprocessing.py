"""Stage 2 - Cleaning, feature/target selection and the train/test split.

Design rule to avoid data leakage: anything that *learns* from data (imputing
medians, scaling means/standard deviations) lives inside the scikit-learn
``Pipeline`` built in ``model_training.py`` and is fitted on the training set
only. This module performs only row-level cleaning that uses no statistics
from other rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.data_loader import DataValidationError


@dataclass(frozen=True)
class CleaningReport:
    """What cleaning changed, so the UI can show it transparently."""

    rows_before: int
    rows_after: int
    missing_target_removed: int
    duplicates_removed: int
    missing_feature_cells: int

    @property
    def rows_removed(self) -> int:
        return self.rows_before - self.rows_after


@dataclass(frozen=True)
class SplitData:
    """Training and testing partitions."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    stratified: bool


def select_features_and_target(
    frame: pd.DataFrame, feature_columns: list[str], target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Return the feature matrix ``X`` and label vector ``y``.

    Raises:
        DataValidationError: If columns are missing, non-numeric or overlap.
    """
    if not feature_columns:
        raise DataValidationError("Select at least one feature column.")
    if target_column in feature_columns:
        raise DataValidationError("The target column cannot also be used as a feature.")

    missing = [column for column in [*feature_columns, target_column] if column not in frame.columns]
    if missing:
        raise DataValidationError(f"Columns not found in the dataset: {', '.join(missing)}")

    non_numeric = [c for c in feature_columns if not pd.api.types.is_numeric_dtype(frame[c])]
    if non_numeric:
        raise DataValidationError(f"Features must be numeric. Non-numeric columns: {', '.join(non_numeric)}")

    return frame[feature_columns].copy(), frame[target_column].copy()


def clean_data(
    frame: pd.DataFrame, feature_columns: list[str], target_column: str
) -> tuple[pd.DataFrame, CleaningReport]:
    """Remove rows that cannot be used for supervised learning.

    * Rows without a label are dropped (there is nothing to learn from).
    * Exact duplicate rows are dropped so the same sample cannot land in both
      the training and the test set (which would inflate test accuracy).

    Missing *feature* values are kept; they are imputed inside the model
    pipeline using training-set medians only.
    """
    columns = [*feature_columns, target_column]
    subset = frame[columns]
    rows_before = len(subset)

    labelled = subset.dropna(subset=[target_column])
    missing_target_removed = rows_before - len(labelled)

    deduplicated = labelled.drop_duplicates()
    duplicates_removed = len(labelled) - len(deduplicated)

    report = CleaningReport(
        rows_before=rows_before,
        rows_after=len(deduplicated),
        missing_target_removed=missing_target_removed,
        duplicates_removed=duplicates_removed,
        missing_feature_cells=int(deduplicated[feature_columns].isna().sum().sum()),
    )
    return deduplicated.reset_index(drop=True), report


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = config.DEFAULT_TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> SplitData:
    """Shuffle and split into training and testing sets.

    The split is *stratified* whenever possible so each class keeps the same
    proportion in both sets. A fixed ``random_state`` makes results reproducible.

    Raises:
        DataValidationError: If ``test_size`` is invalid or the data is too small.
    """
    if not 0.0 < test_size < 1.0:
        raise DataValidationError("The test size must be between 0 and 1.")
    if len(X) != len(y):
        raise DataValidationError("Features and labels must have the same number of rows.")

    n_classes = y.nunique()
    n_test = int(round(len(X) * test_size))
    n_train = len(X) - n_test
    if n_test < n_classes or n_train < n_classes:
        raise DataValidationError(
            "The dataset is too small for this split ratio. Use a different test size or more data."
        )

    class_counts = y.value_counts()
    stratified = bool(class_counts.min() >= 2)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=y if stratified else None,
    )
    return SplitData(X_train, X_test, y_train, y_test, stratified)
