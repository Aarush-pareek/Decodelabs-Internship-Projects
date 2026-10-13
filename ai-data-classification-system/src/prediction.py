"""Stage 5 - Predicting new samples and explaining the result."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.model_training import ModelArtifact


class InvalidInputError(ValueError):
    """Raised when user-entered feature values cannot be used for prediction."""


class PredictionError(RuntimeError):
    """Raised when the model fails to produce a prediction."""


@dataclass
class PredictionResult:
    """Outcome of predicting one sample."""

    label: str
    confidence: float | None  # None if the model cannot output probabilities
    probabilities: dict[str, float]
    warnings: list[str]
    explanation: pd.DataFrame


def validate_input(values: dict[str, object], artifact: ModelArtifact) -> dict[str, float]:
    """Convert raw input to floats and check it against the model's features.

    Raises:
        InvalidInputError: For missing, non-numeric, non-finite or unexpected values.
    """
    expected = artifact.feature_names
    missing = [name for name in expected if name not in values]
    unexpected = [name for name in values if name not in expected]
    if missing:
        raise InvalidInputError(f"Missing values for: {', '.join(missing)}")
    if unexpected:
        raise InvalidInputError(f"Unexpected features: {', '.join(unexpected)}")

    cleaned: dict[str, float] = {}
    for name in expected:
        try:
            number = float(values[name])  # type: ignore[arg-type]
        except (TypeError, ValueError) as error:
            raise InvalidInputError(f"'{name}' must be a number.") from error
        if not math.isfinite(number):
            raise InvalidInputError(f"'{name}' must be a finite number.")
        cleaned[name] = number
    return cleaned


def range_warnings(values: dict[str, float], artifact: ModelArtifact) -> list[str]:
    """Warn when a value lies outside the range seen during training."""
    warnings = []
    for name, value in values.items():
        low, high = artifact.feature_stats.loc[name, "min"], artifact.feature_stats.loc[name, "max"]
        if value < low or value > high:
            warnings.append(
                f"{name} = {value:g} is outside the training range ({low:g} to {high:g}); "
                "the prediction may be less reliable."
            )
    return warnings


def explain_prediction(values: dict[str, float], artifact: ModelArtifact, label: str) -> pd.DataFrame:
    """Compare the sample with the average training sample of the predicted class.

    ``Distance (std)`` is the gap in units of the feature's standard deviation:
    values near 0 mean the sample looks typical for the predicted class.
    For Logistic Regression an extra ``Pull toward class`` column shows how strongly
    each scaled feature pushed the score towards the predicted class.
    """
    stats = artifact.feature_stats
    class_mean = artifact.training_profile.loc[label]
    table = pd.DataFrame(
        {
            "Feature": artifact.feature_names,
            "Your value": [values[name] for name in artifact.feature_names],
            f"Typical '{label}'": [float(class_mean[name]) for name in artifact.feature_names],
        }
    )
    std = stats.loc[artifact.feature_names, "std"].replace(0, np.nan).to_numpy()
    table["Distance (std)"] = np.abs(table["Your value"].to_numpy() - table[f"Typical '{label}'"].to_numpy()) / std
    table["Distance (std)"] = table["Distance (std)"].fillna(0.0)

    classifier = artifact.pipeline.named_steps["classifier"]
    if hasattr(classifier, "coef_"):
        sample = pd.DataFrame([values], columns=artifact.feature_names)
        scaled = artifact.pipeline[:-1].transform(sample)[0]
        class_index = list(artifact.pipeline.classes_).index(label)
        if classifier.coef_.shape[0] > 1:
            coefficients = classifier.coef_[class_index]
        else:  # binary model: one row, expressed for the second class
            coefficients = classifier.coef_[0] if class_index == 1 else -classifier.coef_[0]
        table["Pull toward class"] = scaled * coefficients
    return table


def predict_sample(values: dict[str, object], artifact: ModelArtifact) -> PredictionResult:
    """Validate input, predict the class and build the explanation.

    Raises:
        InvalidInputError: If the input is unusable.
        PredictionError: If the model itself fails.
    """
    cleaned = validate_input(values, artifact)
    sample = pd.DataFrame([cleaned], columns=artifact.feature_names)

    try:
        label = str(artifact.pipeline.predict(sample)[0])
        probabilities: dict[str, float] = {}
        confidence: float | None = None
        if hasattr(artifact.pipeline, "predict_proba"):
            scores = artifact.pipeline.predict_proba(sample)[0]
            probabilities = {str(c): float(p) for c, p in zip(artifact.pipeline.classes_, scores)}
            confidence = probabilities[label]
    except Exception as error:
        raise PredictionError(f"The model could not make a prediction: {error}") from error

    return PredictionResult(
        label=label,
        confidence=confidence,
        probabilities=probabilities,
        warnings=range_warnings(cleaned, artifact),
        explanation=explain_prediction(cleaned, artifact, label),
    )


def explanation_summary(result: PredictionResult) -> str:
    """One short plain-language sentence (markdown) explaining the prediction."""
    table = result.explanation
    confidence = f" with {result.confidence:.1%} confidence" if result.confidence is not None else ""
    text = f"The model predicts **{result.label}**{confidence}. "

    if "Pull toward class" in table.columns:
        strongest = table.loc[table["Pull toward class"].idxmax()]
        text += (
            f"The feature that pushed hardest toward this class was `{strongest['Feature']}` "
            f"(your value: {strongest['Your value']:g}). "
        )
    typical_column = f"Typical '{result.label}'"
    closest = table.loc[table["Distance (std)"].idxmin()]
    text += (
        f"Your sample is most typical of {result.label} in `{closest['Feature']}` "
        f"({closest['Your value']:g} vs. an average of {closest[typical_column]:.2f})."
    )
    return text
