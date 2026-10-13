"""Stage 4 - Model evaluation on the held-out test set."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src import config

METRIC_EXPLANATIONS: dict[str, str] = {
    "Accuracy": "Percentage of all predictions that were correct.",
    "Precision": (
        "Of the samples predicted as a class, how many actually belonged to that class. "
        "High precision means few false alarms."
    ),
    "Recall": (
        "Of the samples that truly belong to a class, how many were found. "
        "High recall means few missed cases."
    ),
    "F1 Score": "The balance (harmonic mean) of precision and recall. It is low if either one is low.",
}

AVERAGING_NOTE = (
    "Precision, recall and F1 are macro-averaged: they are computed for each class and then "
    "averaged, so every class counts equally."
)


def imbalance_warning(class_counts: pd.Series) -> str | None:
    """Explain the 'accuracy mirage' when one class is much rarer than another, else ``None``."""
    smallest, largest = int(class_counts.min()), int(class_counts.max())
    if largest == 0 or smallest / largest >= config.IMBALANCE_RATIO:
        return None
    return (
        f"The classes are imbalanced ({class_counts.idxmin()}: {smallest} samples vs. "
        f"{class_counts.idxmax()}: {largest}). Accuracy alone can look high even when a rare class is "
        "missed, so also check per-class recall and F1 in the classification report."
    )


@dataclass
class EvaluationResult:
    """Metrics for one model on one test set (plain data, safe to save and reload)."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    train_accuracy: float
    class_names: list[str]
    confusion: list[list[int]]
    per_class: list[dict]
    n_test: int

    @property
    def confusion_array(self) -> np.ndarray:
        return np.array(self.confusion)

    @property
    def per_class_table(self) -> pd.DataFrame:
        return pd.DataFrame(self.per_class)

    @property
    def report_table(self) -> pd.DataFrame:
        """Per-class metrics plus an overall macro-average row (a readable classification report)."""
        table = self.per_class_table
        macro = {
            "Class": "Macro average",
            "Precision": table["Precision"].mean(),
            "Recall": table["Recall"].mean(),
            "F1 Score": table["F1 Score"].mean(),
            "Support": int(table["Support"].sum()),
        }
        return pd.concat([table, pd.DataFrame([macro])], ignore_index=True)

    @property
    def generalization_gap(self) -> float:
        """Training accuracy minus test accuracy. A large positive gap suggests overfitting."""
        return self.train_accuracy - self.accuracy

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EvaluationResult":
        return cls(**data)


def evaluate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> EvaluationResult:
    """Score a fitted pipeline on the test set (and training set for comparison)."""
    class_names = [str(label) for label in pipeline.classes_]
    predictions = pipeline.predict(X_test)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, predictions, labels=pipeline.classes_, zero_division=0
    )
    per_class = [
        {
            "Class": name,
            "Precision": float(precision[i]),
            "Recall": float(recall[i]),
            "F1 Score": float(f1[i]),
            "Support": int(support[i]),
        }
        for i, name in enumerate(class_names)
    ]

    return EvaluationResult(
        accuracy=float(accuracy_score(y_test, predictions)),
        precision=float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        recall=float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        f1=float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        train_accuracy=float(accuracy_score(y_train, pipeline.predict(X_train))),
        class_names=class_names,
        confusion=confusion_matrix(y_test, predictions, labels=pipeline.classes_).tolist(),
        per_class=per_class,
        n_test=int(len(y_test)),
    )
