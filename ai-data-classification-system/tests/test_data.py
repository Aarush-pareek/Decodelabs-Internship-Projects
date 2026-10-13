"""Dataset loading, validation and profiling."""

import io

import pandas as pd
import pytest

from src import config
from src.data_loader import (
    DataValidationError,
    Dataset,
    class_distribution,
    data_quality_table,
    load_builtin_iris,
    load_csv,
    summarize_dataset,
    validate_dataset,
)


def csv_bytes(frame: pd.DataFrame) -> io.BytesIO:
    return io.BytesIO(frame.to_csv(index=False).encode())


def make_frame(rows: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {"a": range(rows), "b": [i % 7 for i in range(rows)], "label": ["x", "y"] * (rows // 2)}
    )


def test_builtin_iris_structure(iris):
    assert iris.frame.shape == (150, 5)
    assert iris.target_column == "species"
    assert iris.feature_columns == ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    assert set(iris.frame["species"]) == {"setosa", "versicolor", "virginica"}


def test_sample_csv_matches_builtin_dataset():
    from_csv = load_csv(config.SAMPLE_CSV_PATH, target_column="species")
    pd.testing.assert_frame_equal(from_csv.frame, load_builtin_iris().frame)


def test_summary_and_class_distribution(iris):
    summary = summarize_dataset(iris)
    assert summary["rows"] == 150 and summary["columns"] == 5
    assert summary["features"] == 4 and summary["classes"] == 3
    assert summary["missing_total"] == 0
    assert summary["duplicate_rows"] == 1  # Iris really contains one exact duplicate
    assert class_distribution(iris).to_dict() == {"setosa": 50, "versicolor": 50, "virginica": 50}


def test_data_quality_table_reports_missing_values():
    frame = make_frame()
    frame.loc[0, "a"] = None
    table = data_quality_table(Dataset("t", frame, "label"))
    assert list(table.columns) == ["Column", "Role", "Data type", "Missing", "Missing %", "Unique values"]
    assert table.set_index("Column").loc["a", "Missing"] == 1
    assert table.set_index("Column").loc["label", "Role"] == "Target"


def test_load_csv_from_uploaded_file_object():
    dataset = load_csv(csv_bytes(make_frame()), target_column="label", name="upload.csv")
    assert dataset.name == "upload.csv"
    assert dataset.feature_columns == ["a", "b"]


def test_load_csv_defaults_target_to_last_column():
    assert load_csv(csv_bytes(make_frame())).target_column == "label"


def test_missing_file_gives_friendly_error(tmp_path):
    with pytest.raises(DataValidationError, match="not found"):
        load_csv(tmp_path / "does_not_exist.csv")


def test_empty_file_is_rejected():
    with pytest.raises(DataValidationError, match="empty"):
        load_csv(io.BytesIO(b""))


def test_missing_target_column_is_rejected():
    with pytest.raises(DataValidationError, match="not found"):
        load_csv(csv_bytes(make_frame()), target_column="nope")


def test_dataset_without_numeric_features_is_rejected():
    frame = pd.DataFrame({"text": ["a", "b"] * 20, "label": ["x", "y"] * 20})
    with pytest.raises(DataValidationError, match="numeric"):
        validate_dataset(Dataset("t", frame, "label"))


def test_single_class_is_rejected():
    frame = make_frame()
    frame["label"] = "only"
    with pytest.raises(DataValidationError, match="two different classes"):
        validate_dataset(Dataset("t", frame, "label"))


def test_too_few_rows_is_rejected():
    with pytest.raises(DataValidationError, match="At least"):
        validate_dataset(Dataset("t", make_frame(10), "label"))


def test_class_with_one_sample_is_rejected():
    frame = make_frame()
    frame.loc[0, "label"] = "rare"
    with pytest.raises(DataValidationError, match="rare"):
        validate_dataset(Dataset("t", frame, "label"))


def test_continuous_target_is_rejected():
    frame = make_frame(60)
    frame["label"] = [i * 0.37 for i in range(60)]
    with pytest.raises(DataValidationError, match="distinct values"):
        validate_dataset(Dataset("t", frame, "label"))
