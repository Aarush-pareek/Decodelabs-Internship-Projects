"""Stage 3 - Model construction, training, comparison and persistence.

Every model is wrapped in one scikit-learn ``Pipeline``::

    SimpleImputer(median) -> StandardScaler -> Classifier

Because imputation and scaling are *steps of the pipeline*, they are fitted on
the training data only and are applied identically at prediction time. Saving
the pipeline therefore saves the preprocessing and the model together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src import config


class ModelTrainingError(RuntimeError):
    """Raised when a model cannot be trained, saved or loaded."""


KNN_MODEL = "K-Nearest Neighbors"

MODEL_DESCRIPTIONS: dict[str, str] = {
    "Logistic Regression": (
        "Learns a weight for every feature and converts the weighted sum into class "
        "probabilities. Fast, stable and easy to interpret - the primary model of this project."
    ),
    "Decision Tree": (
        "Learns a sequence of if/else questions about feature values. Easy to visualise, "
        "but a deep tree can memorise the training data."
    ),
    KNN_MODEL: (
        "Classifies a sample by a majority vote of its K closest training samples. "
        "Needs scaled features because it relies on distances."
    ),
}


def available_models() -> list[str]:
    """Names of the supported algorithms (the default comes first)."""
    names = list(MODEL_DESCRIPTIONS)
    names.remove(config.DEFAULT_MODEL)
    return [config.DEFAULT_MODEL, *names]


def _make_classifier(model_name: str, random_state: int):
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=random_state)
    if model_name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=4, random_state=random_state)
    if model_name == KNN_MODEL:
        return KNeighborsClassifier(n_neighbors=config.KNN_NEIGHBORS)
    raise ModelTrainingError(f"Unknown model: '{model_name}'. Choose one of: {', '.join(MODEL_DESCRIPTIONS)}")


def build_pipeline(model_name: str, random_state: int = config.RANDOM_STATE) -> Pipeline:
    """Create an unfitted preprocessing + classifier pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", _make_classifier(model_name, random_state)),
        ]
    )


def train_pipeline(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Fit the pipeline on the training data only and return the fitted copy."""
    fitted = clone(pipeline)
    try:
        fitted.fit(X_train, y_train)
    except Exception as error:  # scikit-learn raises many types; present one clear message
        raise ModelTrainingError(f"Model training failed: {error}") from error
    return fitted


def cross_validated_accuracy(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = config.RANDOM_STATE,
) -> tuple[float, float]:
    """Mean and standard deviation of k-fold accuracy, computed on training data only."""
    folds = min(config.CV_FOLDS, int(y_train.value_counts().min()))
    if folds < 2:
        return float("nan"), float("nan")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(clone(pipeline), X_train, y_train, cv=splitter, scoring="accuracy")
    return float(scores.mean()), float(scores.std())


def knn_k_sweep(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Cross-validated error rate of K-Nearest Neighbors for K = 1, 2, 3, ...

    Uses the training set only (the test set stays untouched). Small K tends to
    overfit noise; large K over-smooths. Returns columns ``K``, ``CV error`` and ``CV std``.
    """
    folds = min(config.CV_FOLDS, int(y_train.value_counts().min()))
    if folds < 2:
        return pd.DataFrame(columns=["K", "CV error", "CV std"])
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    smallest_fold_train = len(X_train) * (folds - 1) // folds
    rows = []
    for k in range(1, min(config.K_SWEEP_MAX, smallest_fold_train) + 1):
        pipeline = build_pipeline(KNN_MODEL, random_state)
        pipeline.set_params(classifier__n_neighbors=k)
        scores = cross_val_score(pipeline, X_train, y_train, cv=splitter, scoring="accuracy")
        rows.append({"K": k, "CV error": float(1 - scores.mean()), "CV std": float(scores.std())})
    return pd.DataFrame(rows)


def recommended_k(sweep: pd.DataFrame) -> int:
    """K with the lowest cross-validated error (ties go to the larger, smoother K)."""
    best = sweep[sweep["CV error"] == sweep["CV error"].min()]
    return int(best["K"].max())


def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Train every supported algorithm and tabulate its performance.

    ``CV accuracy`` uses only the training set, so it is the fair basis for
    *choosing* a model; ``Test accuracy`` reports the final held-out result.
    """
    rows = []
    for name in available_models():
        pipeline = build_pipeline(name, random_state)
        cv_mean, cv_std = cross_validated_accuracy(pipeline, X_train, y_train, random_state)
        fitted = train_pipeline(pipeline, X_train, y_train)
        predictions = fitted.predict(X_test)
        rows.append(
            {
                "Model": name,
                "CV accuracy": cv_mean,
                "CV std": cv_std,
                "Test accuracy": accuracy_score(y_test, predictions),
                "Test F1 (macro)": f1_score(y_test, predictions, average="macro", zero_division=0),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class ModelArtifact:
    """Everything needed to reuse a trained model later, saved as one file."""

    pipeline: Pipeline
    model_name: str
    feature_names: list[str]
    target_name: str
    class_names: list[str]
    test_size: float
    random_state: int
    training_profile: pd.DataFrame  # per-class feature means (raw units) from the training set
    feature_stats: pd.DataFrame  # min / max / mean / std per feature from the training set
    evaluation: dict = field(default_factory=dict)
    dataset_name: str = ""
    trained_at: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))


def build_training_profile(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summaries of the training data used to validate and explain predictions."""
    class_means = X_train.groupby(y_train).mean()
    feature_stats = X_train.agg(["min", "max", "mean", "std"]).T
    return class_means, feature_stats


def save_artifact(artifact: ModelArtifact, path: Path = config.MODEL_PATH) -> Path:
    """Persist the artifact (pipeline + metadata) with joblib."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, path)
    except OSError as error:
        raise ModelTrainingError(f"Could not save the model to '{path.name}': {error}") from error
    return path


def load_artifact(path: Path = config.MODEL_PATH) -> ModelArtifact:
    """Load a previously saved artifact.

    Only load model files you created yourself: joblib files are pickles and can
    execute code when loaded.
    """
    if not path.is_file():
        raise ModelTrainingError("No saved model found yet. Train a model first.")
    try:
        artifact = joblib.load(path)
    except Exception as error:  # corrupted file, version mismatch, ...
        raise ModelTrainingError(f"The saved model could not be loaded: {error}") from error
    if not isinstance(artifact, ModelArtifact):
        raise ModelTrainingError("The file does not contain a model saved by this application.")
    return artifact
