"""Charts for the dashboard.

Every function returns a matplotlib ``Figure`` (no global pyplot state, so no
leaked figures). Design rules: one validated categorical palette assigned to
classes in a fixed order, thin marks, quiet grid, direct value labels, and a
legend whenever more than one series is drawn.
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"

# Categorical palette, fixed order: blue, orange, aqua, yellow, magenta, green, violet, red.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#f3f7fd", "#2a78d6", "#12386b"])
DIVERGING = LinearSegmentedColormap.from_list("div_blue_orange", ["#2a78d6", "#f1f0ed", "#eb6834"])

_RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 10,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "text.color": INK,
}


def class_colors(class_names: list[str]) -> dict[str, str]:
    """Stable class -> colour mapping (order of ``class_names`` decides the colour)."""
    return {name: CATEGORICAL[i % len(CATEGORICAL)] for i, name in enumerate(class_names)}


def _figure(width: float, height: float, **kwargs) -> Figure:
    with mpl.rc_context(_RC):
        return Figure(figsize=(width, height), dpi=110, layout="constrained", **kwargs)


def _quiet_grid(ax, axis: str = "y") -> None:
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_class_distribution(counts: pd.Series, title: str = "Class distribution") -> Figure:
    """Bar chart of samples per class with direct count labels."""
    names = [str(n) for n in counts.index]
    colors = class_colors(names)
    with mpl.rc_context(_RC):
        fig = _figure(5.4, 3.4)
        ax = fig.subplots()
        bars = ax.bar(
            names, counts.to_numpy(), width=0.55, color=[colors[n] for n in names], edgecolor=SURFACE, linewidth=2
        )
        ax.bar_label(bars, padding=3, color=INK, fontsize=10)
        ax.set_title(title)
        ax.set_ylabel("Samples")
        ax.set_ylim(0, counts.max() * 1.15)
        _quiet_grid(ax)
    return fig


def plot_feature_distributions(
    frame: pd.DataFrame, feature_columns: list[str], target_column: str, max_features: int = 8
) -> Figure:
    """Small multiples: one histogram outline per class for each feature."""
    features = feature_columns[:max_features]
    classes = [str(c) for c in sorted(frame[target_column].dropna().unique(), key=str)]
    colors = class_colors(classes)
    labels = frame[target_column].astype(str)

    n_cols = 2 if len(features) > 1 else 1
    n_rows = int(np.ceil(len(features) / n_cols))
    with mpl.rc_context(_RC):
        fig = _figure(5.4 * n_cols, 2.6 * n_rows + 0.6)
        axes = np.atleast_1d(fig.subplots(n_rows, n_cols, squeeze=False)).ravel()
        for ax, feature in zip(axes, features):
            values = frame[feature].dropna()
            bins = np.histogram_bin_edges(values, bins=15)
            for name in classes:
                subset = frame.loc[labels == name, feature].dropna()
                ax.hist(subset, bins=bins, histtype="step", linewidth=2, color=colors[name], label=name)
            ax.set_title(feature.replace("_", " ").title())
            ax.set_ylabel("Samples")
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
            _quiet_grid(ax)
        for ax in axes[len(features):]:
            ax.set_visible(False)
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="outside lower center", ncols=len(classes), title=None)
    return fig


def plot_correlation_heatmap(frame: pd.DataFrame, feature_columns: list[str]) -> Figure:
    """Pearson correlation between numeric features (diverging blue-orange, annotated)."""
    correlation = frame[feature_columns].corr()
    size = max(4.2, 0.9 * len(feature_columns) + 1.8)
    with mpl.rc_context(_RC):
        fig = _figure(size + 0.8, size)
        ax = fig.subplots()
        labels = [c.replace("_", " ") for c in feature_columns]
        sns.heatmap(
            correlation,
            ax=ax,
            cmap=DIVERGING,
            vmin=-1,
            vmax=1,
            center=0,
            annot=len(feature_columns) <= 10,
            fmt=".2f",
            annot_kws={"fontsize": 9, "color": INK},
            linewidths=2,
            linecolor=SURFACE,
            square=True,
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"shrink": 0.8, "label": "Correlation"},
        )
        ax.set_title("Feature correlation")
        ax.tick_params(axis="x", rotation=30)
    return fig


def plot_confusion_matrix(
    matrix: np.ndarray, class_names: list[str], title: str = "Confusion matrix (test set)"
) -> Figure:
    """Counts of true vs predicted classes; the diagonal holds the correct predictions."""
    size = max(4.0, 0.9 * len(class_names) + 2.2)
    with mpl.rc_context(_RC):
        fig = _figure(size + 0.6, size)
        ax = fig.subplots()
        peak = matrix.max() if matrix.size else 1
        image = ax.imshow(matrix, cmap=SEQUENTIAL_BLUE, vmin=0, vmax=max(peak, 1))
        ax.set_xticks(range(len(class_names)), labels=class_names, rotation=30, ha="right")
        ax.set_yticks(range(len(class_names)), labels=class_names)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(title)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = int(matrix[row, col])
                dark_cell = value > 0.55 * max(peak, 1)
                ax.text(col, row, value, ha="center", va="center", fontsize=12, fontweight="bold",
                        color="#ffffff" if dark_cell else INK)
        ax.set_xticks(np.arange(-0.5, len(class_names)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(class_names)), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=3)
        ax.tick_params(which="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.colorbar(image, ax=ax, shrink=0.8, label="Samples")
    return fig


def plot_model_comparison(comparison: pd.DataFrame, selected_model: str | None = None) -> Figure:
    """Grouped bars: cross-validated accuracy (training data) vs test accuracy."""
    models = comparison["Model"].tolist()
    x = np.arange(len(models))
    width = 0.34
    series = [("CV accuracy", "Cross-validated accuracy (training data)"), ("Test accuracy", "Test accuracy")]
    with mpl.rc_context(_RC):
        fig = _figure(6.4, 3.6)
        ax = fig.subplots()
        for offset, (column, label), color in zip((-width / 2, width / 2), series, CATEGORICAL[:2]):
            bars = ax.bar(
                x + offset, comparison[column], width=width, color=color, edgecolor=SURFACE, linewidth=2, label=label
            )
            ax.bar_label(bars, labels=[f"{v:.1%}" for v in comparison[column]], padding=3, fontsize=9, color=INK)
        ax.set_xticks(x, labels=[f"{m}\n(selected)" if m == selected_model else m for m in models])
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Accuracy")
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_title("Model comparison")
        fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncols=2)
        _quiet_grid(ax)
    return fig


def plot_probabilities(probabilities: dict[str, float], predicted: str, class_order: list[str]) -> Figure:
    """Horizontal bars of class probabilities for one prediction."""
    names = [c for c in class_order if c in probabilities]
    values = [probabilities[c] for c in names]
    colors = class_colors(class_order)
    with mpl.rc_context(_RC):
        fig = _figure(5.2, 0.55 * len(names) + 1.0)
        ax = fig.subplots()
        bars = ax.barh(names, values, height=0.55, color=[colors[n] for n in names], edgecolor=SURFACE, linewidth=2)
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in values], padding=4, fontsize=10, color=INK)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.18)
        ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_title("Class probabilities")
        ax.get_yticklabels()[names.index(predicted)].set_fontweight("bold")
        _quiet_grid(ax, axis="x")
    return fig


def plot_k_selection(sweep: pd.DataFrame, used_k: int, best_k: int) -> Figure:
    """Cross-validated error rate against K for K-Nearest Neighbors (the 'elbow' chart)."""
    with mpl.rc_context(_RC):
        fig = _figure(6.4, 3.4)
        ax = fig.subplots()
        color = CATEGORICAL[0]
        ax.plot(sweep["K"], sweep["CV error"], color=color, linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=2)
        if used_k == best_k:
            marks = [(used_k, f"K = {used_k}: used by the model\nand lowest cross-validated error", (30, -44))]
        else:
            marks = [(used_k, f"K = {used_k}: used by the model", (30, -44)),
                     (best_k, f"K = {best_k}: lowest error", (-6, 52))]
        for k, label, offset in marks:
            if k not in set(sweep["K"]):
                continue
            error = float(sweep.loc[sweep["K"] == k, "CV error"].iloc[0])
            ax.scatter([k], [error], s=110, facecolor="none", edgecolor=INK, linewidth=1.6, zorder=3)
            ax.annotate(label, (k, error), textcoords="offset points", xytext=offset, fontsize=9, color=INK,
                        arrowprops={"arrowstyle": "-", "color": INK_SECONDARY, "linewidth": 0.8})
        ax.set_xlabel("K (number of neighbours)")
        ax.set_ylabel("Cross-validated error rate")
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_ylim(bottom=0)
        ax.set_title("Choosing K for K-Nearest Neighbors")
        _quiet_grid(ax)
    return fig
