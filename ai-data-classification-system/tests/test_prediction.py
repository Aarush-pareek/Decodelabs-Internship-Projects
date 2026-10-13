"""Predicting new samples, input validation and explanations."""

import pandas as pd
import pytest

from src.model_training import build_pipeline, build_training_profile, ModelArtifact, train_pipeline
from src.prediction import (
    InvalidInputError,
    explanation_summary,
    predict_sample,
    range_warnings,
    validate_input,
)
from src.preprocessing import select_features_and_target, split_data
from src.workflow import run_experiment


def test_known_samples_are_classified_correctly(artifact, setosa_sample, virginica_sample):
    assert predict_sample(setosa_sample, artifact).label == "setosa"
    assert predict_sample(virginica_sample, artifact).label == "virginica"


def test_prediction_output_types_and_probabilities(artifact, setosa_sample):
    result = predict_sample(setosa_sample, artifact)
    assert isinstance(result.label, str)
    assert 0.0 <= result.confidence <= 1.0
    assert set(result.probabilities) == set(artifact.class_names)
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    assert result.confidence == max(result.probabilities.values())
    assert result.warnings == []


def test_numeric_strings_are_accepted(artifact, setosa_sample):
    as_text = {name: str(value) for name, value in setosa_sample.items()}
    assert predict_sample(as_text, artifact).label == "setosa"


@pytest.mark.parametrize("bad_value", ["abc", None, float("nan"), float("inf"), ""])
def test_invalid_values_are_rejected(artifact, setosa_sample, bad_value):
    setosa_sample["petal_width"] = bad_value
    with pytest.raises(InvalidInputError, match="petal_width"):
        predict_sample(setosa_sample, artifact)


def test_missing_and_unexpected_features_are_rejected(artifact, setosa_sample):
    del setosa_sample["sepal_width"]
    with pytest.raises(InvalidInputError, match="Missing"):
        validate_input(setosa_sample, artifact)
    with pytest.raises(InvalidInputError, match="Unexpected"):
        validate_input({**setosa_sample, "sepal_width": 3.0, "colour": 1}, artifact)


def test_out_of_range_values_trigger_a_warning_but_still_predict(artifact, setosa_sample):
    setosa_sample["sepal_length"] = 25.0
    result = predict_sample(setosa_sample, artifact)
    assert any("sepal_length" in warning for warning in result.warnings)
    assert result.label in artifact.class_names
    in_range = {"sepal_length": 5.0, "sepal_width": 3.0, "petal_length": 3.0, "petal_width": 1.0}
    assert range_warnings(in_range, artifact) == []


def test_explanation_table_structure(artifact, virginica_sample):
    result = predict_sample(virginica_sample, artifact)
    table = result.explanation
    assert len(table) == len(artifact.feature_names)
    assert list(table["Feature"]) == artifact.feature_names
    assert {"Your value", "Distance (std)", "Pull toward class"} <= set(table.columns)
    assert (table["Distance (std)"] >= 0).all()
    assert table["Pull toward class"].sum() > 0  # the evidence favours the predicted class


def test_explanation_summary_mentions_the_predicted_class(artifact, setosa_sample):
    text = explanation_summary(predict_sample(setosa_sample, artifact))
    assert "setosa" in text and "confidence" in text


def test_tree_model_explanation_has_no_coefficient_column(iris, setosa_sample):
    tree = run_experiment(iris, iris.feature_columns, model_name="Decision Tree", save_path=None).artifact
    result = predict_sample(setosa_sample, tree)
    assert result.label == "setosa"
    assert "Pull toward class" not in result.explanation.columns


def test_binary_problem_pull_sign_is_correct_for_both_classes(iris):
    """For two classes sklearn stores one coefficient row; both labels must still be explained correctly."""
    two_class = iris.frame[iris.frame["species"] != "virginica"]
    X, y = select_features_and_target(two_class, iris.feature_columns, "species")
    split = split_data(X, y)
    fitted = train_pipeline(build_pipeline("Logistic Regression"), split.X_train, split.y_train)
    means, stats = build_training_profile(split.X_train, split.y_train)
    artifact = ModelArtifact(fitted, "Logistic Regression", iris.feature_columns, "species", list(fitted.classes_),
                             0.2, 42, means, stats)
    for label in ["setosa", "versicolor"]:
        typical = means.loc[label].to_dict()
        result = predict_sample(typical, artifact)
        assert result.label == label
        assert result.explanation["Pull toward class"].sum() > 0


def test_prediction_uses_the_same_preprocessing_as_training(artifact, setosa_sample):
    """Raw (unscaled) input must give the same answer as pushing it through the stored pipeline manually."""
    sample = pd.DataFrame([setosa_sample])
    expected = artifact.pipeline.predict(sample)[0]
    assert predict_sample(setosa_sample, artifact).label == expected
