"""Every chart builds and renders to an image without error."""

import io

import numpy as np
import pytest
from matplotlib.figure import Figure

from src.data_loader import class_distribution
from src.model_training import recommended_k
from utils import visualization as viz


def render(figure) -> None:
    assert isinstance(figure, Figure)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    assert buffer.getbuffer().nbytes > 1000


def test_dataset_charts(iris):
    render(viz.plot_class_distribution(class_distribution(iris)))
    render(viz.plot_feature_distributions(iris.frame, iris.feature_columns, iris.target_column))
    render(viz.plot_correlation_heatmap(iris.frame, iris.feature_columns))


def test_single_feature_distribution_does_not_crash(iris):
    render(viz.plot_feature_distributions(iris.frame, ["petal_length"], iris.target_column))


def test_result_charts(experiment, artifact):
    evaluation = experiment.evaluation
    render(viz.plot_confusion_matrix(evaluation.confusion_array, evaluation.class_names))
    render(viz.plot_model_comparison(experiment.comparison, artifact.model_name))
    render(viz.plot_probabilities({"setosa": 0.9, "versicolor": 0.08, "virginica": 0.02}, "setosa",
                                  artifact.class_names))


def test_k_selection_chart_for_same_and_different_k(experiment):
    best = recommended_k(experiment.k_sweep)
    render(viz.plot_k_selection(experiment.k_sweep, best, best))
    render(viz.plot_k_selection(experiment.k_sweep, 9, best))


def test_class_colors_are_stable_and_distinct():
    colors = viz.class_colors(["a", "b", "c"])
    assert len(set(colors.values())) == 3
    assert viz.class_colors(["a", "b", "c"]) == colors


@pytest.mark.parametrize("count", [2, 8])
def test_confusion_matrix_handles_different_class_counts(count):
    render(viz.plot_confusion_matrix(np.eye(count, dtype=int) * 3, [f"c{i}" for i in range(count)]))
