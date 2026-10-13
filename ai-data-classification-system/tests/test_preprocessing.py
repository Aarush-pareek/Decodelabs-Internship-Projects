"""Cleaning, feature selection and the train/test split."""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataValidationError
from src.model_training import build_pipeline, train_pipeline
from src.preprocessing import clean_data, select_features_and_target, split_data

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def test_select_features_and_target_shapes(iris):
    X, y = select_features_and_target(iris.frame, FEATURES, "species")
    assert X.shape == (150, 4) and len(y) == 150
    assert list(X.columns) == FEATURES


@pytest.mark.parametrize(
    "features, target, message",
    [
        ([], "species", "at least one"),
        (["species"], "species", "cannot also be"),
        (["nope"], "species", "not found"),
    ],
)
def test_select_features_rejects_bad_selection(iris, features, target, message):
    with pytest.raises(DataValidationError, match=message):
        select_features_and_target(iris.frame, features, target)


def test_select_features_rejects_non_numeric_feature():
    frame = pd.DataFrame({"a": ["x", "y"], "b": [1, 2], "label": [0, 1]})
    with pytest.raises(DataValidationError, match="numeric"):
        select_features_and_target(frame, ["a"], "label")


def test_clean_data_removes_duplicates_and_unlabelled_rows(iris):
    frame = iris.frame.copy()
    frame.loc[0, "species"] = None
    cleaned, report = clean_data(frame, FEATURES, "species")
    assert report.missing_target_removed == 1
    assert report.duplicates_removed == 1
    assert report.rows_before == 150 and report.rows_after == len(cleaned) == 148
    assert report.rows_removed == 2
    assert cleaned["species"].notna().all()


def test_clean_data_keeps_missing_feature_values_for_the_pipeline(iris):
    frame = iris.frame.copy()
    frame.loc[3, "petal_width"] = np.nan
    cleaned, report = clean_data(frame, FEATURES, "species")
    assert report.missing_feature_cells == 1
    assert cleaned["petal_width"].isna().sum() == 1


def test_split_sizes_and_stratification(iris):
    X, y = select_features_and_target(iris.frame, FEATURES, "species")
    split = split_data(X, y, test_size=0.2)
    assert len(split.X_train) == 120 and len(split.X_test) == 30
    assert split.stratified
    assert split.y_test.value_counts().to_dict() == {"setosa": 10, "versicolor": 10, "virginica": 10}


def test_train_and_test_sets_do_not_overlap(iris):
    X, y = select_features_and_target(iris.frame, FEATURES, "species")
    split = split_data(X, y)
    assert set(split.X_train.index).isdisjoint(split.X_test.index)
    assert len(set(split.X_train.index) | set(split.X_test.index)) == len(X)


def test_split_is_reproducible_and_seed_dependent(iris):
    X, y = select_features_and_target(iris.frame, FEATURES, "species")
    first = split_data(X, y, random_state=7)
    again = split_data(X, y, random_state=7)
    other = split_data(X, y, random_state=8)
    assert list(first.X_test.index) == list(again.X_test.index)
    assert list(first.X_test.index) != list(other.X_test.index)


@pytest.mark.parametrize("bad_size", [0, 1, -0.2, 1.5])
def test_split_rejects_invalid_test_size(iris, bad_size):
    X, y = select_features_and_target(iris.frame, FEATURES, "species")
    with pytest.raises(DataValidationError, match="between 0 and 1"):
        split_data(X, y, test_size=bad_size)


def test_split_rejects_tiny_dataset():
    X = pd.DataFrame({"a": range(6)})
    y = pd.Series([0, 0, 0, 1, 1, 1])
    with pytest.raises(DataValidationError, match="too small"):
        split_data(X, y, test_size=0.1)


def test_scaler_learns_from_training_data_only(iris):
    """Data-leakage check: the fitted scaler must reflect the training set, not the whole dataset."""
    X, y = select_features_and_target(iris.frame, FEATURES, "species")
    split = split_data(X, y)
    fitted = train_pipeline(build_pipeline("Logistic Regression"), split.X_train, split.y_train)
    scaler_means = fitted.named_steps["scaler"].mean_
    np.testing.assert_allclose(scaler_means, split.X_train.mean().to_numpy())
    assert not np.allclose(scaler_means, X.mean().to_numpy())
