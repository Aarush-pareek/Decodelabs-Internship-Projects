"""Central configuration: paths and default settings.

All paths are resolved relative to the project root so the project runs
unchanged on any machine and any operating system.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"

SAMPLE_CSV_PATH: Path = DATA_DIR / "iris.csv"
MODEL_PATH: Path = MODELS_DIR / "trained_model.pkl"

# Reproducibility: the same seed always produces the same split and model.
RANDOM_STATE: int = 42

# Train/test split settings (the user can change the ratio in the UI).
DEFAULT_TEST_SIZE: float = 0.20
MIN_TEST_SIZE: float = 0.10
MAX_TEST_SIZE: float = 0.50

# Cross-validation folds used when comparing models on the training set.
CV_FOLDS: int = 5

# Smallest dataset the pipeline will accept (needs data to split and learn from).
MIN_ROWS: int = 20
MIN_ROWS_PER_CLASS: int = 2
MAX_CLASSES: int = 8  # charts use a fixed 8-colour categorical palette

DEFAULT_MODEL: str = "Logistic Regression"

# K-Nearest Neighbors: neighbours used by the model, and the largest K tried in the K-selection chart.
KNN_NEIGHBORS: int = 5
K_SWEEP_MAX: int = 25

# A class smaller than this fraction of the largest class makes accuracy misleading.
IMBALANCE_RATIO: float = 0.5

# Dataset names offered in the UI.
DATASET_IRIS_BUILTIN: str = "Iris (built-in)"
DATASET_IRIS_CSV: str = "Iris (CSV sample)"
DATASET_UPLOAD: str = "Upload your own CSV"
