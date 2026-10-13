"""Model construction, training, evaluation, persistence and the end-to-end workflow."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src import config
from src.evaluation import EvaluationResult, evaluate_model, imbalance_warning
from src.model_training import (
    ModelArtifact,
    ModelTrainingError,
    available_models,
    build_pipeline,
    compare_models,
    cross_validated_accuracy,
    knn_k_sweep,
    load_artifact,
    recommended_k,
    save_artifact,
    train_pipeline,
)
from src.preprocessing import select_features_and_target, split_data
from src.workflow import run_experiment


@pytest.fixture(scope="module")
def split(iris):
    X, y = select_features_and_target(iris.frame, iris.feature_columns, iris.target_column)
    return split_data(X, y)


def test_default_model_is_logistic_regression():
    assert available_models()[0] == config.DEFAULT_MODEL == "Logistic Regression"
    assert set(available_models()) == {"Logistic Regression", "Decision Tree", "K-Nearest Neighbors"}


def test_pipeline_has_preprocessing_before_classifier():
    pipeline = build_pipeline("Logistic Regression")
    assert [name for name, _ in pipeline.steps] == ["imputer", "scaler", "classifier"]


def test_unknown_model_raises():
    with pytest.raises(ModelTrainingError, match="Unknown model"):
        build_pipeline("Magic Forest")


@pytest.mark.parametrize("model_name", ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors"])
def test_every_model_trains_and_beats_chance(model_name, split):
    fitted = train_pipeline(build_pipeline(model_name), split.X_train, split.y_train)
    predictions = fitted.predict(split.X_test)
    assert predictions.shape == (len(split.X_test),)
    assert accuracy_score(split.y_test, predictions) > 0.85


def test_training_does_not_modify_the_unfitted_pipeline(split):
    pipeline = build_pipeline("Logistic Regression")
    train_pipeline(pipeline, split.X_train, split.y_train)
    assert not hasattr(pipeline.named_steps["classifier"], "coef_")


def test_pipeline_imputes_missing_feature_values(split):
    fitted = train_pipeline(build_pipeline("Logistic Regression"), split.X_train, split.y_train)
    sample = split.X_test.iloc[[0]].copy()
    sample.iloc[0, 0] = np.nan
    assert fitted.predict(sample).shape == (1,)


def test_training_failure_becomes_model_training_error(split):
    one_class = pd.Series(["setosa"] * len(split.y_train), index=split.y_train.index)
    with pytest.raises(ModelTrainingError, match="training failed"):
        train_pipeline(build_pipeline("Logistic Regression"), split.X_train, one_class)


def test_cross_validated_accuracy_uses_training_data_only(split):
    mean, std = cross_validated_accuracy(build_pipeline("Logistic Regression"), split.X_train, split.y_train)
    assert 0.8 < mean <= 1.0 and std >= 0


def test_compare_models_returns_one_row_per_algorithm(split):
    table = compare_models(split.X_train, split.y_train, split.X_test, split.y_test)
    assert list(table["Model"]) == available_models()
    assert table[["CV accuracy", "Test accuracy"]].ge(0).all().all()


def test_evaluation_metrics_match_scikit_learn(split):
    fitted = train_pipeline(build_pipeline("Logistic Regression"), split.X_train, split.y_train)
    result = evaluate_model(fitted, split.X_train, split.y_train, split.X_test, split.y_test)
    predictions = fitted.predict(split.X_test)
    assert result.accuracy == pytest.approx(accuracy_score(split.y_test, predictions))
    assert result.precision == pytest.approx(precision_score(split.y_test, predictions, average="macro"))
    assert result.recall == pytest.approx(recall_score(split.y_test, predictions, average="macro"))
    assert result.f1 == pytest.approx(f1_score(split.y_test, predictions, average="macro"))
    assert result.confusion_array.shape == (3, 3)
    np.testing.assert_array_equal(result.confusion_array, confusion_matrix(split.y_test, predictions))
    assert result.confusion_array.sum() == result.n_test == len(split.y_test)


def test_evaluation_report_table_and_roundtrip(experiment):
    evaluation = experiment.evaluation
    report = evaluation.report_table
    assert list(report["Class"]) == [*evaluation.class_names, "Macro average"]
    assert report.iloc[-1]["Support"] == evaluation.n_test
    assert report.iloc[-1]["F1 Score"] == pytest.approx(evaluation.f1)
    restored = EvaluationResult.from_dict(evaluation.to_dict())
    assert restored.accuracy == evaluation.accuracy
    np.testing.assert_array_equal(restored.confusion_array, evaluation.confusion_array)


def test_iris_logistic_regression_performance_is_sensible(experiment):
    assert experiment.evaluation.accuracy >= 0.90
    assert experiment.cv_mean >= 0.90
    assert abs(experiment.evaluation.generalization_gap) < 0.10


def test_run_experiment_is_deterministic(iris):
    first = run_experiment(iris, iris.feature_columns, save_path=None)
    second = run_experiment(iris, iris.feature_columns, save_path=None)
    assert first.evaluation.accuracy == second.evaluation.accuracy
    np.testing.assert_array_equal(first.evaluation.confusion_array, second.evaluation.confusion_array)


def test_run_experiment_reports_cleaning_and_comparison(experiment):
    assert experiment.cleaning.duplicates_removed == 1
    assert len(experiment.split.X_train) + len(experiment.split.X_test) == 149
    assert experiment.comparison is not None and len(experiment.comparison) == 3


def test_save_and_load_roundtrip_gives_identical_predictions(experiment, tmp_path):
    path = save_artifact(experiment.artifact, tmp_path / "nested" / "model.pkl")
    assert path.is_file()
    restored = load_artifact(path)
    assert isinstance(restored, ModelArtifact)
    assert restored.feature_names == experiment.artifact.feature_names
    X_test = experiment.split.X_test
    np.testing.assert_array_equal(restored.pipeline.predict(X_test), experiment.artifact.pipeline.predict(X_test))


def test_saved_pipeline_contains_the_preprocessing_steps(experiment, tmp_path):
    restored = load_artifact(save_artifact(experiment.artifact, tmp_path / "model.pkl"))
    assert "scaler" in restored.pipeline.named_steps and "imputer" in restored.pipeline.named_steps


def test_load_missing_model_raises(tmp_path):
    with pytest.raises(ModelTrainingError, match="No saved model"):
        load_artifact(tmp_path / "missing.pkl")


def test_load_corrupted_model_raises(tmp_path):
    broken = tmp_path / "broken.pkl"
    broken.write_bytes(b"this is not a joblib file")
    with pytest.raises(ModelTrainingError, match="could not be loaded"):
        load_artifact(broken)


def test_load_foreign_object_raises(tmp_path):
    import joblib

    path = tmp_path / "other.pkl"
    joblib.dump({"not": "an artifact"}, path)
    with pytest.raises(ModelTrainingError, match="not contain a model"):
        load_artifact(path)


def test_knn_uses_five_neighbours_by_default():
    assert build_pipeline("K-Nearest Neighbors").named_steps["classifier"].n_neighbors == config.KNN_NEIGHBORS == 5


def test_knn_k_sweep_structure_and_values(split):
    sweep = knn_k_sweep(split.X_train, split.y_train)
    assert list(sweep.columns) == ["K", "CV error", "CV std"]
    assert list(sweep["K"]) == list(range(1, config.K_SWEEP_MAX + 1))
    assert sweep["CV error"].between(0, 1).all()
    assert sweep["CV error"].min() < 0.10  # Iris is easy for KNN
    assert 1 <= recommended_k(sweep) <= config.K_SWEEP_MAX


def test_recommended_k_prefers_the_larger_k_on_ties():
    sweep = pd.DataFrame({"K": [1, 2, 3, 4], "CV error": [0.2, 0.1, 0.3, 0.1], "CV std": [0.0] * 4})
    assert recommended_k(sweep) == 4


def test_knn_k_sweep_is_limited_by_dataset_size():
    X = pd.DataFrame({"a": range(24)})
    y = pd.Series(["x", "y"] * 12)
    sweep = knn_k_sweep(X, y)
    assert sweep["K"].max() <= 24 * 4 // 5


def test_run_experiment_returns_k_sweep_only_when_relevant(iris):
    assert run_experiment(iris, iris.feature_columns, save_path=None).k_sweep is None
    knn = run_experiment(iris, iris.feature_columns, model_name="K-Nearest Neighbors", save_path=None)
    assert knn.k_sweep is not None and len(knn.k_sweep) == config.K_SWEEP_MAX


def test_imbalance_warning_only_for_skewed_classes():
    assert imbalance_warning(pd.Series({"a": 50, "b": 50, "c": 50})) is None
    assert imbalance_warning(pd.Series({"a": 50, "b": 30})) is None
    message = imbalance_warning(pd.Series({"common": 95, "rare": 5}))
    assert message is not None and "rare" in message and "recall" in message


def test_accuracy_can_hide_a_missed_rare_class():
    """The 'accuracy mirage': high accuracy, yet one class has poor recall - per-class metrics expose it."""
    import numpy as np
    rng = np.random.default_rng(0)
    common = pd.DataFrame({"a": rng.normal(0, 1, 190), "b": rng.normal(0, 1, 190)})
    rare = pd.DataFrame({"a": rng.normal(0.5, 1, 10), "b": rng.normal(0.5, 1, 10)})
    X = pd.concat([common, rare], ignore_index=True)
    y = pd.Series(["common"] * 190 + ["rare"] * 10)
    tiny = split_data(X, y)
    fitted = train_pipeline(build_pipeline("Logistic Regression"), tiny.X_train, tiny.y_train)
    result = evaluate_model(fitted, tiny.X_train, tiny.y_train, tiny.X_test, tiny.y_test)
    per_class = result.per_class_table.set_index("Class")
    assert result.accuracy >= 0.90
    assert per_class.loc["rare", "Recall"] < per_class.loc["common", "Recall"]
