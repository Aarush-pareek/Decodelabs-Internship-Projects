"""AI Data Classification System - Streamlit dashboard.

Run with:  streamlit run app.py

This file only handles the user interface. All machine-learning logic lives in
``src/`` and all charts in ``utils/visualization.py``.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from src import config
from src.data_loader import (
    DataValidationError,
    Dataset,
    class_distribution,
    data_quality_table,
    load_builtin_iris,
    load_csv,
    read_csv_frame,
    summarize_dataset,
    validate_dataset,
)
from src.evaluation import AVERAGING_NOTE, METRIC_EXPLANATIONS, EvaluationResult, imbalance_warning
from src.model_training import (
    MODEL_DESCRIPTIONS,
    ModelArtifact,
    ModelTrainingError,
    available_models,
    load_artifact,
    recommended_k,
    save_artifact,
)
from src.prediction import InvalidInputError, PredictionError, PredictionResult, explanation_summary, predict_sample
from src.preprocessing import CleaningReport, clean_data
from src.workflow import run_experiment
from utils import visualization as viz
from utils.html_components import CUSTOM_CSS, metric_card, prediction_card, section_header, status_pill

logger = logging.getLogger(__name__)

PIPELINE_STAGES = [
    "Load dataset",
    "Explore data",
    "Clean and preprocess",
    "Select features and target",
    "Train / test split",
    "Train model",
    "Evaluate",
    "Predict new data",
]

STATE_KEY = "model_state"
PREDICTION_KEY = "last_prediction"


@dataclass
class ModelState:
    """Everything the dashboard needs to show a trained (or loaded) model."""

    artifact: ModelArtifact
    evaluation: EvaluationResult
    origin: str  # "trained" or "loaded"
    cleaning: CleaningReport | None = None
    cv_scores: tuple[float, float] | None = None
    comparison: pd.DataFrame | None = None
    k_sweep: pd.DataFrame | None = None
    n_train: int | None = None
    save_error: str | None = None


@dataclass
class Controls:
    """Values chosen in the sidebar."""

    dataset: Dataset | None
    model_name: str
    compare: bool
    train_clicked: bool
    load_clicked: bool


# --------------------------------------------------------------------------- helpers


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def cards(items: list[tuple[str, str, str]]) -> None:
    """Render a row of metric cards: (label, value, hint)."""
    for column, (label, value, hint) in zip(st.columns(len(items)), items):
        with column:
            html(metric_card(label, value, hint))


def feature_key(dataset: Dataset) -> str:
    return f"features::{dataset.name}::{dataset.target_column}"


def split_key(dataset: Dataset) -> str:
    return f"test_size::{dataset.name}::{dataset.target_column}"


def selected_features(dataset: Dataset) -> list[str]:
    return list(st.session_state.get(feature_key(dataset), dataset.feature_columns))


def selected_test_size(dataset: Dataset) -> float:
    return st.session_state.get(split_key(dataset), int(config.DEFAULT_TEST_SIZE * 100)) / 100


def current_state() -> ModelState | None:
    return st.session_state.get(STATE_KEY)


def pretty(name: str) -> str:
    return name.replace("_", " ").title()


def percent(value: float) -> str:
    return f"{value:.1%}"


# --------------------------------------------------------------------------- sidebar


def render_sidebar() -> Controls:
    """Project info, dataset selection, model selection and the action buttons."""
    with st.sidebar:
        st.markdown("### AI Data Classification System")
        st.caption("Internship Project 2 - Data Classification Using AI")
        with st.expander("Pipeline stages"):
            st.markdown("\n".join(f"{i}. {stage}" for i, stage in enumerate(PIPELINE_STAGES, start=1)))
        st.divider()

        st.markdown("**Dataset**")
        source = st.radio(
            "Dataset source",
            [config.DATASET_IRIS_BUILTIN, config.DATASET_IRIS_CSV, config.DATASET_UPLOAD],
            label_visibility="collapsed",
        )
        dataset = _load_selected_dataset(source)

        st.markdown("**Model**")
        model_name = st.selectbox(
            "Algorithm",
            available_models(),
            index=0,
            label_visibility="collapsed",
            help=f"{config.DEFAULT_MODEL} is the default model of this project.",
        )
        compare = st.checkbox(
            "Compare all algorithms",
            value=True,
            help="Also trains Decision Tree and K-Nearest Neighbors for a side-by-side comparison.",
        )
        st.divider()

        train_clicked = st.button("Train model", type="primary", disabled=dataset is None)
        load_clicked = st.button("Load saved model", disabled=not config.MODEL_PATH.is_file())
        st.caption("A trained model is saved to `models/trained_model.pkl`.")

    return Controls(dataset, model_name, compare, train_clicked, load_clicked)


def _load_selected_dataset(source: str) -> Dataset | None:
    """Load the dataset chosen in the sidebar, showing a friendly message on failure."""
    try:
        if source == config.DATASET_IRIS_BUILTIN:
            return load_builtin_iris()
        if source == config.DATASET_IRIS_CSV:
            return load_csv(config.SAMPLE_CSV_PATH, target_column="species", name=config.DATASET_IRIS_CSV)
        return _load_uploaded_dataset()
    except DataValidationError as error:
        st.error(str(error))
    except Exception:  # last line of defence: never show a traceback for bad input
        logger.exception("Unexpected error while loading the dataset")
        st.error("The dataset could not be loaded. Please check the file and try again.")
    return None


def _load_uploaded_dataset() -> Dataset | None:
    upload = st.file_uploader("CSV file", type=["csv"], help="Numeric feature columns and one label column.")
    if upload is None:
        st.caption("Upload a CSV file to begin.")
        return None
    frame = read_csv_frame(io.BytesIO(upload.getvalue()))
    target = st.selectbox("Target column (class labels)", list(frame.columns), index=len(frame.columns) - 1)
    dataset = Dataset(name=upload.name, frame=frame, target_column=target)
    validate_dataset(dataset)
    return dataset


# --------------------------------------------------------------------------- actions


def handle_actions(controls: Controls) -> None:
    """Run training / loading when the sidebar buttons were pressed."""
    dataset = controls.dataset
    if dataset is None:
        return
    if controls.train_clicked:
        _train(dataset, controls)
    elif controls.load_clicked:
        _load_saved()


def _train(dataset: Dataset, controls: Controls) -> None:
    try:
        with st.spinner(f"Training {controls.model_name}..."):
            experiment = run_experiment(
                dataset,
                selected_features(dataset),
                model_name=controls.model_name,
                test_size=selected_test_size(dataset),
                compare=controls.compare,
                save_path=None,
            )
    except (DataValidationError, ModelTrainingError) as error:
        st.error(str(error))
        return
    except Exception:
        logger.exception("Unexpected training failure")
        st.error("Training failed unexpectedly. Check the dataset and selected options, then try again.")
        return

    save_error = None
    try:
        save_artifact(experiment.artifact)
    except ModelTrainingError as error:
        save_error = str(error)  # keep the trained model usable even if saving fails

    st.session_state[STATE_KEY] = ModelState(
        artifact=experiment.artifact,
        evaluation=experiment.evaluation,
        origin="trained",
        cleaning=experiment.cleaning,
        cv_scores=(experiment.cv_mean, experiment.cv_std),
        comparison=experiment.comparison,
        k_sweep=experiment.k_sweep,
        n_train=len(experiment.split.X_train),
        save_error=save_error,
    )
    st.session_state.pop(PREDICTION_KEY, None)


def _load_saved() -> None:
    try:
        artifact = load_artifact()
        evaluation = EvaluationResult.from_dict(artifact.evaluation)
    except (ModelTrainingError, TypeError, KeyError) as error:
        st.error(f"The saved model could not be used: {error}")
        return
    st.session_state[STATE_KEY] = ModelState(artifact=artifact, evaluation=evaluation, origin="loaded")
    st.session_state.pop(PREDICTION_KEY, None)


# --------------------------------------------------------------------------- sections


def render_header() -> None:
    st.title("AI Data Classification System")
    st.caption("A supervised-learning workflow: explore data, train a classifier, evaluate it and make predictions.")


def render_overview(dataset: Dataset, model_name: str, state: ModelState | None) -> None:
    html(section_header(1, "Project Overview", "A snapshot of the selected dataset and model."))
    summary = summarize_dataset(dataset)
    shown_model = state.artifact.model_name if state else model_name
    if state is None:
        status_text, status_kind = "Not trained", "idle"
    elif state.origin == "loaded":
        status_text, status_kind = "Loaded from disk", "ok"
    else:
        status_text, status_kind = "Trained", "ok"

    cards(
        [
            ("Dataset size", f"{summary['rows']:,} rows", f"{summary['columns']} columns in total"),
            ("Features", str(summary["features"]), "Numeric input columns"),
            ("Classes", str(summary["classes"]), f"Target: {dataset.target_column}"),
            ("Algorithm", shown_model, "Selected classifier"),
        ]
    )
    html(f'<div style="margin-top:.7rem">Model status&nbsp; {status_pill(status_text, status_kind)}</div>')


def render_explorer(dataset: Dataset) -> None:
    html(section_header(2, "Dataset Explorer", "Understand the data before modelling it."))
    summary = summarize_dataset(dataset)
    cards(
        [
            ("Rows", f"{summary['rows']:,}", ""),
            ("Columns", str(summary["columns"]), ""),
            ("Missing values", str(summary["missing_total"]), "Across all columns"),
            ("Duplicate rows", str(summary["duplicate_rows"]), "Removed during preprocessing"),
        ]
    )
    st.write("")
    tabs = st.tabs(["Preview", "Statistics", "Data quality", "Class balance", "Feature distributions", "Correlation"])
    with tabs[0]:
        st.dataframe(dataset.frame.head(20), hide_index=True)
        st.caption("First 20 rows.")
    with tabs[1]:
        st.dataframe(dataset.frame.describe().T.round(3))
        st.caption("Count, mean, standard deviation, minimum, quartiles and maximum for each numeric column.")
    with tabs[2]:
        st.dataframe(data_quality_table(dataset), hide_index=True)
        st.caption("Data types, missing values and unique values for every column.")
    with tabs[3]:
        st.pyplot(viz.plot_class_distribution(class_distribution(dataset)))
        counts = class_distribution(dataset)
        st.caption(f"Largest class: {counts.max()} samples, smallest class: {counts.min()} samples.")
    with tabs[4]:
        st.pyplot(viz.plot_feature_distributions(dataset.frame, dataset.feature_columns, dataset.target_column))
        st.caption("Well-separated curves mean that a feature helps tell the classes apart.")
    with tabs[5]:
        if len(dataset.feature_columns) >= 2:
            st.pyplot(viz.plot_correlation_heatmap(dataset.frame, dataset.feature_columns))
            st.caption("Values near +1 or -1 mean two features carry very similar information.")
        else:
            st.info("At least two numeric features are needed for a correlation heatmap.")


def render_preprocessing(dataset: Dataset) -> None:
    html(section_header(3, "Data Preprocessing", "Choose the model inputs and how the data is split."))
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.multiselect(
            "Feature columns (model inputs)",
            options=dataset.feature_columns,
            default=dataset.feature_columns,
            key=feature_key(dataset),
        )
        html(metric_card("Target variable", dataset.target_column, "The class the model learns to predict"))
        st.write("")
        st.slider(
            "Test set size",
            min_value=int(config.MIN_TEST_SIZE * 100),
            max_value=int(config.MAX_TEST_SIZE * 100),
            value=int(config.DEFAULT_TEST_SIZE * 100),
            step=5,
            format="%d%%",
            key=split_key(dataset),
            help="Share of the data kept aside to test the model. It is never used for training.",
        )

    with right:
        st.markdown("**What happens when you train**")
        features = selected_features(dataset)
        if not features:
            st.warning("Select at least one feature column.")
            return
        cleaned, report = clean_data(dataset.frame, features, dataset.target_column)
        test_size = selected_test_size(dataset)
        n_test = round(len(cleaned) * test_size)
        st.markdown(
            f"1. **Clean:** remove {report.missing_target_removed} unlabelled and {report.duplicates_removed} "
            f"duplicate row(s) - {report.rows_after} rows remain.\n"
            f"2. **Split:** {report.rows_after - n_test} training rows and {n_test} testing rows "
            f"({test_size:.0%} test), shuffled and stratified so every class keeps its share. "
            f"Random seed: {config.RANDOM_STATE}.\n"
            "3. **Pipeline:** fill missing values with the median, standardise features "
            "(mean 0, standard deviation 1), then classify. Both steps learn from the training rows only."
        )
        if report.missing_feature_cells:
            st.info(f"{report.missing_feature_cells} missing feature value(s) will be filled with training medians.")


def stale_model_warning(state: ModelState, dataset: Dataset) -> None:
    artifact = state.artifact
    if artifact.dataset_name != dataset.name or artifact.target_name != dataset.target_column:
        st.warning(
            f"The model shown was trained on '{artifact.dataset_name}'. "
            "Press **Train model** to train on the current dataset."
        )


def render_training(dataset: Dataset, state: ModelState | None) -> None:
    html(section_header(4, "Model Training", "Fit the selected algorithm on the training set."))
    if state is None:
        st.info("No model yet. Choose an algorithm in the sidebar and press **Train model**.")
        return

    stale_model_warning(state, dataset)
    evaluation = state.evaluation
    cards(
        [
            ("Algorithm", state.artifact.model_name, f"Trained {state.artifact.trained_at}"),
            ("Training status", "Completed" if state.origin == "trained" else "Loaded", state.origin.capitalize()),
            ("Training accuracy", percent(evaluation.train_accuracy), "On data the model learned from"),
            ("Testing accuracy", percent(evaluation.accuracy), f"On {evaluation.n_test} unseen samples"),
        ]
    )
    if state.save_error:
        st.warning(f"The model works for this session but could not be saved. {state.save_error}")
    elif state.origin == "trained":
        st.success("Model trained and saved to `models/trained_model.pkl` (preprocessing and model together).")

    st.write("")
    st.caption(MODEL_DESCRIPTIONS[state.artifact.model_name])
    gap = evaluation.generalization_gap
    if gap > 0.10:
        st.warning(
            f"Training accuracy is {gap:.1%} higher than testing accuracy. "
            "The model may be overfitting (memorising the training data)."
        )
    else:
        st.caption(
            f"Training and testing accuracy differ by {abs(gap):.1%}, "
            "which suggests the model generalises to unseen data."
        )
    if state.cv_scores and not pd.isna(state.cv_scores[0]):
        st.caption(
            f"5-fold cross-validation on the training set: {state.cv_scores[0]:.1%} "
            f"(+/- {state.cv_scores[1]:.1%}). This is a more stable estimate than a single split."
        )

    if state.comparison is not None:
        st.markdown("**Algorithm comparison**")
        left, right = st.columns([3, 2], gap="large")
        with left:
            st.pyplot(viz.plot_model_comparison(state.comparison, state.artifact.model_name))
        with right:
            st.dataframe(
                state.comparison.rename(columns={"CV std": "CV spread"}),
                hide_index=True,
                column_config={
                    column: st.column_config.NumberColumn(format="%.3f")
                    for column in ["CV accuracy", "CV spread", "Test accuracy", "Test F1 (macro)"]
                },
            )
            st.caption("Models are compared by cross-validation on training data, so the test set stays untouched.")

    if state.k_sweep is not None and not state.k_sweep.empty:
        with st.expander("Choosing K for K-Nearest Neighbors"):
            st.pyplot(viz.plot_k_selection(state.k_sweep, config.KNN_NEIGHBORS, recommended_k(state.k_sweep)))
            st.caption(
                "K is the number of neighbours that vote on a class. A very small K follows noise (overfitting); "
                "a very large K over-smooths (underfitting). The best K is where the cross-validated error is lowest."
            )


def render_evaluation(dataset: Dataset, state: ModelState | None) -> None:
    html(section_header(5, "Model Evaluation", "How well does the model perform on data it has never seen?"))
    if state is None:
        st.info("Train or load a model to see its evaluation.")
        return

    evaluation = state.evaluation
    cards(
        [
            ("Accuracy", percent(evaluation.accuracy), METRIC_EXPLANATIONS["Accuracy"]),
            ("Precision", percent(evaluation.precision), "Of the predictions for a class, how many were right."),
            ("Recall", percent(evaluation.recall), "Of the real members of a class, how many were found."),
            ("F1 Score", percent(evaluation.f1), METRIC_EXPLANATIONS["F1 Score"]),
        ]
    )
    st.caption(AVERAGING_NOTE)
    imbalance = imbalance_warning(class_distribution(dataset))
    if imbalance:
        st.warning(imbalance)

    st.write("")
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.pyplot(viz.plot_confusion_matrix(evaluation.confusion_array, evaluation.class_names))
    with right:
        st.markdown("**Classification report**")
        st.dataframe(
            evaluation.report_table,
            hide_index=True,
            column_config={
                column: st.column_config.NumberColumn(format="%.3f") for column in ["Precision", "Recall", "F1 Score"]
            },
        )
        st.markdown("**How to read the confusion matrix**")
        st.markdown(
            "Each row is the *true* class and each column the *predicted* class. "
            "Numbers on the diagonal are correct predictions; any number off the diagonal is a mistake "
            "where the model confused one class with another."
        )
        errors = int(evaluation.confusion_array.sum() - evaluation.confusion_array.trace())
        st.caption(f"{errors} of {evaluation.n_test} test samples were misclassified.")

    with st.expander("What do these metrics mean?"):
        for metric, explanation in METRIC_EXPLANATIONS.items():
            st.markdown(f"**{metric}** - {explanation}")


def render_prediction(state: ModelState | None) -> None:
    html(section_header(6, "Prediction", "Enter feature values and let the trained model classify the sample."))
    if state is None:
        st.info("Train or load a model to make predictions.")
        return

    artifact = state.artifact
    stats = artifact.feature_stats
    values: dict[str, float] = {}
    with st.form("prediction_form"):
        columns = st.columns(2)
        for index, name in enumerate(artifact.feature_names):
            low, high, mean, std = (float(stats.loc[name, key]) for key in ("min", "max", "mean", "std"))
            step = float(f"{std / 20:.1g}") if std and std > 0 else 0.1
            with columns[index % 2]:
                values[name] = st.number_input(
                    pretty(name),
                    value=round(mean, 2),
                    step=step,
                    format="%.2f",
                    key=f"input::{artifact.trained_at}::{name}",
                    help=f"Training range: {low:g} to {high:g}",
                )
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        try:
            st.session_state[PREDICTION_KEY] = predict_sample(values, artifact)
        except (InvalidInputError, PredictionError) as error:
            st.session_state.pop(PREDICTION_KEY, None)
            st.error(str(error))
        except Exception:
            logger.exception("Unexpected prediction failure")
            st.session_state.pop(PREDICTION_KEY, None)
            st.error("The prediction failed unexpectedly. Please check the entered values.")

    result: PredictionResult | None = st.session_state.get(PREDICTION_KEY)
    if result is not None:
        _render_prediction_result(result, artifact)


def _render_prediction_result(result: PredictionResult, artifact: ModelArtifact) -> None:
    colors = viz.class_colors(artifact.class_names)
    for warning in result.warnings:
        st.warning(warning)

    left, right = st.columns([1, 1], gap="large")
    with left:
        html(prediction_card(result.label, result.confidence, colors.get(result.label, viz.CATEGORICAL[0])))
    with right:
        if result.probabilities:
            st.pyplot(viz.plot_probabilities(result.probabilities, result.label, artifact.class_names))

    st.markdown("**Why this prediction?**")
    st.markdown(explanation_summary(result))
    columns = {"Distance (std)": st.column_config.NumberColumn(format="%.2f")}
    if "Pull toward class" in result.explanation.columns:
        columns["Pull toward class"] = st.column_config.NumberColumn(format="%.2f")
    st.dataframe(result.explanation, hide_index=True, column_config=columns)
    st.caption(
        "Distance (std): how far your value is from the average of the predicted class, measured in standard "
        "deviations (smaller = more typical). Pull toward class: how strongly each standardised feature increased "
        "the model's score for the predicted class (Logistic Regression only)."
    )


# --------------------------------------------------------------------------- entry point


def main() -> None:
    st.set_page_config(page_title="AI Data Classification System", layout="wide")
    html(CUSTOM_CSS)

    controls = render_sidebar()
    render_header()

    dataset = controls.dataset
    if dataset is None:
        st.info("Select a valid dataset in the sidebar to begin.")
        return

    handle_actions(controls)
    state = current_state()

    render_overview(dataset, controls.model_name, state)
    render_explorer(dataset)
    render_preprocessing(dataset)
    render_training(dataset, state)
    render_evaluation(dataset, state)
    render_prediction(state)


main()
