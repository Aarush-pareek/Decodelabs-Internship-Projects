"""Shared fixtures. Every test works on temporary files, never on the real models/ folder."""

import pytest

from src.data_loader import load_builtin_iris
from src.workflow import run_experiment


@pytest.fixture(scope="session")
def iris():
    return load_builtin_iris()


@pytest.fixture(scope="session")
def experiment(iris, tmp_path_factory):
    """One complete training run (Logistic Regression + comparison), saved to a temp file."""
    path = tmp_path_factory.mktemp("models") / "model.pkl"
    return run_experiment(iris, iris.feature_columns, compare=True, save_path=path)


@pytest.fixture(scope="session")
def artifact(experiment):
    return experiment.artifact


@pytest.fixture
def setosa_sample():
    return {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}


@pytest.fixture
def virginica_sample():
    return {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.8, "petal_width": 2.2}
