"""Runs the full supervised-learning pipeline from dataset to saved model.

Stages (each implemented in its own module):
    clean -> select features -> split -> train -> evaluate -> package -> save
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src import config
from src.data_loader import Dataset
from src.evaluation import EvaluationResult, evaluate_model
from src.model_training import (
    ModelArtifact,
    build_pipeline,
    build_training_profile,
    compare_models,
    cross_validated_accuracy,
    KNN_MODEL,
    knn_k_sweep,
    save_artifact,
    train_pipeline,
)
from src.preprocessing import CleaningReport, SplitData, clean_data, select_features_and_target, split_data


@dataclass
class ExperimentResult:
    """All outputs of one training run."""

    artifact: ModelArtifact
    evaluation: EvaluationResult
    cleaning: CleaningReport
    split: SplitData
    cv_mean: float
    cv_std: float
    comparison: pd.DataFrame | None
    k_sweep: pd.DataFrame | None
    saved_path: Path | None


def run_experiment(
    dataset: Dataset,
    feature_columns: list[str],
    model_name: str = config.DEFAULT_MODEL,
    test_size: float = config.DEFAULT_TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
    compare: bool = False,
    save_path: Path | None = config.MODEL_PATH,
) -> ExperimentResult:
    """Train and evaluate one model on a dataset.

    Raises:
        DataValidationError: For unusable data or selections.
        ModelTrainingError: If training or saving fails.
    """
    cleaned, cleaning = clean_data(dataset.frame, feature_columns, dataset.target_column)
    X, y = select_features_and_target(cleaned, feature_columns, dataset.target_column)
    split = split_data(X, y, test_size=test_size, random_state=random_state)

    pipeline = build_pipeline(model_name, random_state)
    cv_mean, cv_std = cross_validated_accuracy(pipeline, split.X_train, split.y_train, random_state)
    fitted = train_pipeline(pipeline, split.X_train, split.y_train)
    evaluation = evaluate_model(fitted, split.X_train, split.y_train, split.X_test, split.y_test)

    comparison = (
        compare_models(split.X_train, split.y_train, split.X_test, split.y_test, random_state) if compare else None
    )

    k_sweep = None
    if compare or model_name == KNN_MODEL:
        k_sweep = knn_k_sweep(split.X_train, split.y_train, random_state)

    class_means, feature_stats = build_training_profile(split.X_train, split.y_train)
    artifact = ModelArtifact(
        pipeline=fitted,
        model_name=model_name,
        feature_names=list(feature_columns),
        target_name=dataset.target_column,
        class_names=evaluation.class_names,
        test_size=test_size,
        random_state=random_state,
        training_profile=class_means,
        feature_stats=feature_stats,
        evaluation=evaluation.to_dict(),
        dataset_name=dataset.name,
    )
    saved_path = save_artifact(artifact, save_path) if save_path is not None else None

    return ExperimentResult(artifact, evaluation, cleaning, split, cv_mean, cv_std, comparison, k_sweep, saved_path)
