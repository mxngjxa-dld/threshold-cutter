from __future__ import annotations

import io
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import average_precision_score, precision_recall_curve

from app.utils.activations import apply_activation
from app.utils.data_io import (
    ColumnSelection,
    DataMetadata,
    get_majority_class,
    infer_column_candidates,
    prepare_score_matrix,
    validate_data,
)
from app.utils.exports import (
    generate_timestamp,
    save_classification_report,
    save_confusion_matrix_image,
    save_optimal_metrics,
    save_predictions_csv,
    save_roc_images,
)
from app.utils.metrics import MetricSummary, create_metrics_summary
from app.utils.plots import (
    create_inline_roc_display,
    plot_confusion_matrix_raw,
    plot_confusion_matrix_norm,
    # plot_combined_roc_all_models,   # ← new function
)
from app.utils.thresholds import (
    ThresholdResult,
    compute_optimal_thresholds_youden,
    predict_with_thresholds,
)

APP_TITLE = "Multiclass Threshold Tuner"
ASSETS_DIR = Path(__file__).parent / "app/assets"
STYLESHEET = ASSETS_DIR / "styles.css"
ACTIVATION_OPTIONS = ["none", "softmax", "sigmoid", "sigmoid_5"]


@st.cache_data(show_spinner=False)
def parse_uploaded_csv(file_bytes: bytes, delimiter: str | None) -> pd.DataFrame:
    if not file_bytes:
        raise ValueError("Uploaded file is empty.")
    buffer = io.BytesIO(file_bytes)
    if delimiter:
        df = pd.read_csv(buffer, sep=delimiter)
    else:
        df = pd.read_csv(buffer, sep=None, engine="python")
    if df.empty:
        raise ValueError("Loaded CSV is empty; please provide a dataset with rows.")
    return df


@st.cache_data(show_spinner=False)
def prepare_dataset(
    df: pd.DataFrame,
    selection: ColumnSelection | None,
) -> dict[str, object]:
    """
    Validate raw dataframe and return core artefacts for downstream processing.
    """
    metadata = validate_data(df, column_selection=selection)
    scores, index = prepare_score_matrix(
        df,
        metadata.classes,
        metadata,
        return_index=True,
    )
    y_true = _align_true_labels(df, metadata, index)
    fallback_class = get_majority_class(y_true)
    return {
        "metadata": metadata,
        "scores": scores,
        "index": index,
        "y_true": y_true,
        "fallback_class": fallback_class,
    }


@st.cache_data(show_spinner=False)
def compute_predictions_and_metrics(
    scores: np.ndarray,
    y_true: np.ndarray,
    classes: tuple[str, ...],
    activation: str,
    thresholds_signature: tuple[tuple[str, float], ...],
    fallback_class: str,
) -> dict[str, object]:
    """
    Execute activation, thresholding, and metrics computation under caching.
    """
    thresholds_map = {cls: float(val) for cls, val in thresholds_signature}
    result = predict_with_thresholds(
        scores,
        classes,
        thresholds=thresholds_map,
        activation=activation,
        fallback_class=fallback_class,
    )
    summary = create_metrics_summary(
        y_true=y_true,
        y_pred=result.predictions,
        y_scores=result.activated_scores,
        classes=classes,
    )
    optimal = compute_optimal_thresholds_youden(
        y_true=list(y_true),  # Convert ndarray to list/Sequence
        y_scores=result.activated_scores,
        classes=classes,
    )
    return {
        "result": result,
        "summary": summary,
        "optimal_thresholds": optimal,
    }


def inject_css(path: Path) -> None:
    if path.exists():
        css = path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def ensure_session_defaults() -> None:
    defaults = {
        "global_threshold": 0.5,
        "auto_global": False,
        "auto_per_class": False,
        "class_thresholds": {},
        "class_filter": [],
        "activation_prev": None,
        "dataset_signature": None,
        "fallback_class": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_default_threshold(activation: str) -> float:
    activation = (activation or "none").lower()
    if activation in {"sigmoid", "sigmoid_5", "softmax"}:
        return 0.5
    return 0.0


def get_threshold_bounds(activation: str) -> tuple[float, float, float]:
    activation = (activation or "none").lower()
    if activation in {"sigmoid", "sigmoid_5", "softmax"}:
        return 0.0, 1.0, 0.01
    return 0.0, 40.0, 0.1


def _align_true_labels(
    df: pd.DataFrame,
    metadata: DataMetadata,
    index: pd.Index,
) -> np.ndarray:
    label_column = metadata.true_label_column
    if label_column not in df.columns:
        raise KeyError(
            f"True label column '{label_column}' not found in uploaded data."
        )

    if metadata.format == "wide":
        aligned = df.loc[index, label_column]
        return aligned.to_numpy()

    id_col = metadata.sample_id_column
    if id_col and id_col in df.columns:
        label_series = (
            df[[id_col, label_column]]
            .drop_duplicates(subset=id_col, keep="first")
            .set_index(id_col)[label_column]
        )
        aligned = label_series.reindex(index)
        return aligned.to_numpy()

    series = df[label_column].reset_index(drop=True)
    if len(series) == len(index):
        return series.to_numpy()

    try:
        positions = index.astype(int)
        return series.iloc[positions].to_numpy()
    except (TypeError, ValueError, AttributeError):
        return series.to_numpy()


def _build_threshold_signature(
    class_thresholds: Mapping[str, float],
    classes: Sequence[str],
) -> tuple[tuple[str, float], ...]:
    return tuple((cls, float(class_thresholds[cls])) for cls in classes)


def _compute_auto_threshold_map(
    optimal_df: pd.DataFrame,
    classes: Sequence[str],
    default_value: float,
) -> dict[str, float]:
    auto_map: dict[str, float] = {}
    for cls in classes:
        value = (
            float(optimal_df.loc[cls, "optimal_threshold"])
            if cls in optimal_df.index
            else float("nan")
        )
        if not np.isfinite(value):
            value = default_value
        auto_map[cls] = value
    return auto_map


def _build_prediction_dataframe(
    sample_index: pd.Index,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    activated_scores: np.ndarray,
    classes: Sequence[str],
    activation: str,
    class_thresholds: Mapping[str, float],
    global_threshold: float,
    fallback_class: str,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "sample_key": sample_index,
            "true_label": y_true,
            "predicted_class": y_pred,
        }
    )
    classes = list(sorted(classes))
    for idx, cls in enumerate(classes):
        df[f"score_{cls}"] = activated_scores[:, idx]
    for cls, value in class_thresholds.items():
        df[f"threshold_{cls}"] = value
    df["global_threshold"] = float(global_threshold)
    df["activation"] = activation
    df["fallback_class"] = fallback_class
    return df


def _render_metric_cards(summary: MetricSummary) -> None:
    st.markdown(
        """
            <div class="main-card metric-row">
            <div class="metric-card">
                <h3>Accuracy</h3>
                <div class="value">{accuracy:.4f}</div>
            </div>
            <div class="metric-card">
                <h3>Macro Precision</h3>
                <div class="value">{precision:.4f}</div>
            </div>
            <div class="metric-card">
                <h3>Macro Recall</h3>
                <div class="value">{recall:.4f}</div>
            </div>
            <div class="metric-card">
                <h3>Macro F1</h3>
                <div class="value">{f1:.4f}</div>
            </div>
            </div>
        """.format(
                    accuracy=summary.accuracy,
                    precision=summary.macro_precision,
                    recall=summary.macro_recall,
                    f1=summary.macro_f1,
                ),
                unsafe_allow_html=True,
            )


def _render_youden_table(
    summary: MetricSummary,
    selected_classes: Sequence[str],
) -> None:
    display_df = summary.per_class.loc[
        selected_classes, ["optimal_threshold", "youden_j", "tpr", "fpr", "auc"]
    ]
    display_df = display_df.rename(
        columns={
            "optimal_threshold": "optimal_threshold",
            "youden_j": "youden_j",
            "tpr": "tpr",
            "fpr": "fpr",
            "auc": "auc",
        }
    )
    display_df = display_df.reset_index().rename(columns={"index": "class"})
    st.markdown(
        display_df.to_html(
            index=False,
            classes="youden-table",
            float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "NaN",
        ),
        unsafe_allow_html=True,
    )


def _build_pr_curve_figure(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    classes: Sequence[str],
    filter_classes: Sequence[str],
) -> plt.Figure | None:
    chosen = [cls for cls in classes if cls in filter_classes]
    if not chosen:
        return None

    has_curve = False
    fig, ax = plt.subplots(figsize=(8, 6))
    for cls in chosen:
        idx = classes.index(cls)
        binary_true = (y_true == cls).astype(int)
        if binary_true.sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(binary_true, y_scores[:, idx])
        ap = average_precision_score(binary_true, y_scores[:, idx])
        ax.step(recall, precision, where="post", label=f"{cls} (AP={ap:.3f})")
        has_curve = True

    if not has_curve:
        plt.close(fig)
        return None

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def _handle_downloads(
    summary: MetricSummary,
    optimal_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    thresholds_map: Mapping[str, float],
    classes: Sequence[str],
    confusion_fig, 
    confusion_fig_norm,
    roc_figures: Mapping[str, plt.Figure],
) -> None:
    st.sidebar.markdown("### Downloads")
    timestamp = generate_timestamp()

    if st.sidebar.button("Save metrics summary"):
        overall = {
            "accuracy": summary.accuracy,
            "macro_precision": summary.macro_precision,
            "macro_recall": summary.macro_recall,
            "macro_f1": summary.macro_f1,
            "micro_auc": summary.micro_auc,
            "macro_auc": summary.macro_auc,
        }
        per_class = summary.per_class.reset_index().to_dict(orient="records")
        youden = optimal_df.reset_index().to_dict(orient="records")
        report = {
            "overall": overall,
            "per_class": per_class,
            "youden": youden,
        }
        paths = save_classification_report(report, timestamp=timestamp)
        st.sidebar.success(f"Metrics saved: {paths['json'].name}, {paths['csv'].name}")

    if st.sidebar.button("Save optimal metrics"):
        paths = save_optimal_metrics(optimal_df, timestamp=timestamp)
        st.sidebar.success(
            f"Optimal metrics saved: {paths['json'].name}, {paths['csv'].name}"
        )

    if st.sidebar.button("Save confusion matrix image"):
        path = save_confusion_matrix_image(
            confusion_fig, 
            norm=False, 
            timestamp=timestamp
            )
        path = save_confusion_matrix_image(
            confusion_fig_norm, 
            norm=True, 
            timestamp=timestamp
            )
        st.sidebar.success(f"Confusion matrices saved: {path.name}")

    if st.sidebar.button("Save ROC figures"):
        paths = save_roc_images(dict(roc_figures), classes, timestamp=timestamp)
        st.sidebar.success(
            f"Saved ROC images ({len(paths)} files) to {paths[next(iter(paths))].parent}"
        )

    if st.sidebar.button("Save predictions CSV"):
        path = save_predictions_csv(
            predictions_df,
            thresholds=dict(thresholds_map),
            timestamp=timestamp,
        )
        st.sidebar.success(f"Predictions saved: {path.name}")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    inject_css(STYLESHEET)
    ensure_session_defaults()

    st.title(APP_TITLE)
    st.caption(
        "Interactive Streamlit application for multiclass threshold tuning and evaluation."
    )

    with st.sidebar:
        st.header("Dataset")
        uploaded_file = st.file_uploader(
            "Upload classification data (CSV)",
            type=["csv"],
            accept_multiple_files=False,
        )
        delimiter_input = st.text_input(
            "Delimiter override",
            help="Leave blank for auto-detection.",
        )
        delimiter = delimiter_input or None
        st.divider()
        st.header("Activation")
        default_activation = st.session_state["activation_prev"] or "none"
        activation = st.selectbox(
            "Activation function",
            options=ACTIVATION_OPTIONS,
            index=ACTIVATION_OPTIONS.index(default_activation),
        )
        st.session_state["activation_prev"] = activation

    if uploaded_file is None:
        st.info("Upload a CSV file to begin threshold tuning.")
        return

    try:
        df = parse_uploaded_csv(uploaded_file.getvalue(), delimiter)
    except Exception as exc:
        st.error(f"Failed to read CSV: {exc}")
        return

    dataset_signature = (tuple(df.columns), len(df))
    if st.session_state["dataset_signature"] != dataset_signature:
        st.session_state["dataset_signature"] = dataset_signature
        st.session_state["class_thresholds"] = {}
        st.session_state["class_filter"] = []
        st.session_state["fallback_class"] = None
        st.session_state.pop("column_mapping_signature", None)
        for key in list(st.session_state.keys()):
            if key.startswith("threshold_") or key.startswith("column_mapping_"):
                del st.session_state[key]

    try:
        column_candidates = infer_column_candidates(df)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to infer column mappings: {exc}")
        return

    format_options = ["wide", "long"]
    stored_format = st.session_state.get(
        "column_mapping_format", column_candidates.default_format
    )
    if stored_format not in format_options:
        stored_format = column_candidates.default_format

    true_label_options = list(column_candidates.true_label_options)
    if not true_label_options:
        true_label_options = list(df.columns)

    with st.sidebar:
        st.header("Column Mapping")
        format_choice = st.radio(
            "Data layout",
            options=format_options,
            index=format_options.index(stored_format),
            key="column_mapping_format",
            help="Select whether the uploaded data is organised in wide or long format.",
        )

        true_label_default = st.session_state.get(
            "column_mapping_true_label", column_candidates.true_label_default
        )
        if true_label_default not in true_label_options:
            true_label_default = true_label_options[0]
        true_label_index = true_label_options.index(true_label_default)
        true_label_column = st.selectbox(
            "True label column",
            options=true_label_options,
            index=true_label_index,
            key="column_mapping_true_label",
            help="Column containing ground-truth labels.",
        )

        remaining_columns = [col for col in df.columns if col != true_label_column]

        sample_id_default = st.session_state.get(
            "column_mapping_sample_id",
            column_candidates.sample_id_default or "<None>",
        )
        sample_id_options = ["<None>"] + remaining_columns
        if sample_id_default not in sample_id_options:
            sample_id_default = "<None>"
        sample_id_index = sample_id_options.index(sample_id_default)
        sample_id_choice = st.selectbox(
            "Sample identifier column",
            options=sample_id_options,
            index=sample_id_index,
            key="column_mapping_sample_id",
            help="Optional column used to align long-format rows belonging to the same sample.",
        )
        sample_id_column = None if sample_id_choice == "<None>" else sample_id_choice

        long_class_column: str | None = None
        long_score_column: str | None = None
        wide_columns: list[str] = []

        if format_choice == "wide":
            preferred_wide = [
                col
                for col in column_candidates.wide_score_options
                if col in remaining_columns
            ]
            fallback_wide = [
                col for col in remaining_columns if col not in preferred_wide
            ]
            wide_options = preferred_wide + fallback_wide
            if "column_mapping_wide_scores" in st.session_state:
                current_wide = [
                    col
                    for col in st.session_state["column_mapping_wide_scores"]
                    if col in wide_options
                ]
            else:
                current_wide = list(column_candidates.wide_score_default)
            wide_columns = st.multiselect(
                "Score columns",
                options=wide_options,
                default=current_wide,
                key="column_mapping_wide_scores",
                help="Numeric score columns used to build the per-class score matrix.",
            )
            if len(wide_columns) < 2:
                st.info("Select at least two score columns for wide-format datasets.")
        else:
            preferred_class = [
                col
                for col in column_candidates.long_class_options
                if col in remaining_columns
            ]
            fallback_class = [
                col for col in remaining_columns if col not in preferred_class
            ]
            long_class_options = preferred_class + fallback_class
            if not long_class_options:
                long_class_options = remaining_columns
            if (
                "column_mapping_long_class" in st.session_state
                and st.session_state["column_mapping_long_class"]
                not in long_class_options
            ):
                del st.session_state["column_mapping_long_class"]
            long_class_default = st.session_state.get(
                "column_mapping_long_class",
                column_candidates.long_class_default
                or (long_class_options[0] if long_class_options else None),
            )
            if long_class_default not in long_class_options and long_class_options:
                long_class_default = long_class_options[0]
            long_class_index = (
                long_class_options.index(long_class_default)
                if long_class_default in long_class_options
                else 0
            )
            long_class_column = st.selectbox(
                "Predicted class column",
                options=long_class_options,
                index=long_class_index,
                key="column_mapping_long_class",
                help="Column providing the predicted class per row.",
            )

            score_candidates = [
                col
                for col in column_candidates.long_score_options
                if col in remaining_columns
            ]
            fallback_score = [
                col for col in remaining_columns if col not in score_candidates
            ]
            long_score_options = [
                col
                for col in score_candidates + fallback_score
                if col != long_class_column
            ]
            if not long_score_options:
                long_score_options = [
                    col for col in remaining_columns if col != long_class_column
                ]
            if not long_score_options:
                st.error(
                    "Unable to determine a score column; please add a numeric score column to the dataset."
                )
                return
            if (
                "column_mapping_long_score" in st.session_state
                and st.session_state["column_mapping_long_score"]
                not in long_score_options
            ):
                del st.session_state["column_mapping_long_score"]
            long_score_default = st.session_state.get(
                "column_mapping_long_score",
                column_candidates.long_score_default or long_score_options[0],
            )
            if long_score_default not in long_score_options:
                long_score_default = long_score_options[0]
            long_score_index = long_score_options.index(long_score_default)
            long_score_column = st.selectbox(
                "Score column",
                options=long_score_options,
                index=long_score_index,
                key="column_mapping_long_score",
                help="Numeric score for the predicted class.",
            )

    column_selection = ColumnSelection(
        format=format_choice,
        true_label=true_label_column,
        wide_score_columns=tuple(wide_columns) if format_choice == "wide" else (),
        long_class_column=long_class_column if format_choice == "long" else None,
        long_score_column=long_score_column if format_choice == "long" else None,
        sample_id_column=sample_id_column,
    )

    mapping_signature = (
        column_selection.format,
        column_selection.true_label,
        column_selection.sample_id_column,
        column_selection.long_class_column,
        column_selection.long_score_column,
        column_selection.wide_score_columns,
    )
    if st.session_state.get("column_mapping_signature") != mapping_signature:
        st.session_state["column_mapping_signature"] = mapping_signature
        st.session_state["class_thresholds"] = {}
        st.session_state["class_filter"] = []
        st.session_state["fallback_class"] = None
        for key in list(st.session_state.keys()):
            if key.startswith("threshold_"):
                del st.session_state[key]

    with st.spinner("Preparing dataset..."):
        try:
            data_bundle = prepare_dataset(df, column_selection)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Data validation failed: {exc}")
            return

    metadata: DataMetadata = data_bundle["metadata"]  # type: ignore[assignment]
    classes = list(metadata.classes)
    scores: np.ndarray = data_bundle["scores"]  # type: ignore[assignment]
    y_true: np.ndarray = data_bundle["y_true"]  # type: ignore[assignment]
    sample_index: pd.Index = data_bundle["index"]  # type: ignore[assignment]
    fallback_default = data_bundle["fallback_class"]  # type: ignore[assignment]

    default_threshold = get_default_threshold(activation)
    st.session_state.setdefault("global_threshold", default_threshold)

    if st.session_state["fallback_class"] not in classes:
        st.session_state["fallback_class"] = (
            fallback_default if fallback_default in classes else classes[0]
        )

    class_thresholds_state: dict[str, float] = st.session_state["class_thresholds"]
    for cls in classes:
        class_thresholds_state.setdefault(cls, default_threshold)
    for cls in list(class_thresholds_state.keys()):
        if cls not in classes:
            class_thresholds_state.pop(cls, None)

    with st.sidebar:
        st.header("Thresholds")
        auto_global = st.checkbox(
            "Auto global threshold",
            key="auto_global",
            help="Derive a single threshold from Youden's J averages.",
        )
        auto_per_class = st.checkbox(
            "Auto per-class thresholds",
            key="auto_per_class",
            help="Use class-wise Youden-optimal thresholds.",
        )
        min_thr, max_thr, step_thr = get_threshold_bounds(activation)
        if (
            "class_filter" not in st.session_state
            or not st.session_state["class_filter"]
        ):
            st.session_state["class_filter"] = list(classes)

        st.header("Fallback & Filtering")
        fallback_idx = classes.index(st.session_state["fallback_class"])
        fallback_class = st.selectbox(
            "Fallback class",
            options=classes,
            index=fallback_idx,
            key="fallback_class",
            help="Class assigned when no thresholds are exceeded.",
        )

    activated_scores = apply_activation(scores, activation)
    optimal_thresholds_df = compute_optimal_thresholds_youden(
        y_true,
        activated_scores,
        classes,
    )
    auto_threshold_map = _compute_auto_threshold_map(
        optimal_thresholds_df,
        classes,
        default_threshold,
    )
    finite_values = [v for v in auto_threshold_map.values() if np.isfinite(v)]
    auto_global_value = (
        float(np.mean(finite_values)) if finite_values else default_threshold
    )

    if auto_global:
        st.session_state["global_threshold"] = auto_global_value
    if (
        st.session_state["global_threshold"] < min_thr
        or st.session_state["global_threshold"] > max_thr
    ):
        st.session_state["global_threshold"] = float(
            np.clip(st.session_state["global_threshold"], min_thr, max_thr)
        )

    with st.sidebar:
        global_threshold = st.slider(
            "Global threshold",
            min_value=float(min_thr),
            max_value=float(max_thr),
            value=float(st.session_state["global_threshold"]),
            step=float(step_thr),
            key="global_threshold",
            disabled=auto_global,
        )
        apply_global_threshold = st.checkbox(
            label="Apply global threshold",
            key="apply_global_threshold",
            help="Use the global value for every class",
        )

        st.markdown("### Per-class thresholds")
        per_class_thresholds: dict[str, float] = {}
        for cls in list(sorted(classes)):
            key = f"threshold_{cls}"
            if auto_per_class:
                value = auto_threshold_map.get(cls, global_threshold)
                st.session_state[key] = value
            elif apply_global_threshold:
                value = st.session_state.global_threshold
                st.session_state[key] = value
            else:
                st.session_state.setdefault(
                    key, class_thresholds_state.get(cls, global_threshold)
                )
            number_value = st.number_input(
                f"{cls}",
                min_value=float(min_thr),
                max_value=float(max_thr),
                value=float(st.session_state[key]),
                step=float(step_thr),
                key=key,
                disabled=auto_per_class or apply_global_threshold,
                format="%.3f" 
            )

            class_thresholds_state[cls] = float(number_value)
            per_class_thresholds[cls] = float(number_value)

        st.markdown("#### Class filter")
        if any(cls not in classes for cls in st.session_state["class_filter"]):
            st.session_state["class_filter"] = list(classes)
        selected_classes = st.multiselect(
            "Classes to visualize",
            options=classes,
            key="class_filter",
        )
        if not selected_classes:
            selected_classes = classes

    thresholds_signature = _build_threshold_signature(per_class_thresholds, classes)

    computation = compute_predictions_and_metrics(
        scores=scores,
        y_true=y_true,
        classes=tuple(classes),
        activation=activation,
        thresholds_signature=thresholds_signature,
        fallback_class=st.session_state["fallback_class"],
    )
    result: ThresholdResult = computation["result"]  # type: ignore[assignment]
    summary: MetricSummary = computation["summary"]  # type: ignore[assignment]
    optimal_thresholds_df = computation["optimal_thresholds"]  # type: ignore[assignment]

    predictions_df = _build_prediction_dataframe(
        sample_index=sample_index,
        y_true=y_true,
        y_pred=result.predictions,
        activated_scores=result.activated_scores,
        classes=classes,
        activation=activation,
        class_thresholds=per_class_thresholds,
        global_threshold=st.session_state["global_threshold"],
        fallback_class=st.session_state["fallback_class"],
    )

    st.subheader("Evaluation Overview")
    _render_metric_cards(summary)

    micro_auc_text = (
        f"{summary.micro_auc:.4f}" if summary.micro_auc is not None else "N/A"
    )
    macro_auc_text = (
        f"{summary.macro_auc:.4f}" if summary.macro_auc is not None else "N/A"
    )
    st.markdown(
        f"**Micro AUC:** {micro_auc_text} &nbsp;&nbsp;·&nbsp;&nbsp; **Macro AUC:** {macro_auc_text}"
    )

    filter_mask = np.isin(y_true, selected_classes)
    filtered_true = y_true[filter_mask]
    filtered_pred = result.predictions[filter_mask]
    confusion_fig = plot_confusion_matrix_raw(
        filtered_true,
        filtered_pred,
        classes=selected_classes,
        f1_macro=summary.macro_f1,
    )
    confusion_fig_norm = plot_confusion_matrix_norm(
        filtered_true,
        filtered_pred,
        classes=selected_classes,
        f1_macro=summary.macro_f1,
    )
    st.markdown("### Confusion Matrix")
    st.pyplot(confusion_fig, width="content")
    plt.close(confusion_fig)

    st.markdown("### Normalized Confusion Matrix")
    st.pyplot(confusion_fig_norm, width="content")
    plt.close(confusion_fig_norm)

    st.markdown("### ROC Curves")
    roc_figures = create_inline_roc_display(
        roc_df=summary.per_class,
        classes=classes,
        filter_classes=selected_classes,
    )
    if "combined" in roc_figures:
        st.pyplot(roc_figures["combined"], width="content")
    for cls, fig in roc_figures.items():
        if cls == "combined":
            continue
        st.pyplot(fig, width="content")
    for fig in roc_figures.values():
        plt.close(fig)

    st.markdown("### Youden's J Statistics")
    _render_youden_table(summary, selected_classes)

    st.markdown("### Precision-Recall Curves")
    pr_fig = _build_pr_curve_figure(
        y_true=y_true,
        y_scores=result.activated_scores,
        classes=classes,
        filter_classes=selected_classes,
    )
    if pr_fig is not None:
        st.pyplot(pr_fig, width="content")
        plt.close(pr_fig)
    else:
        st.info(
            "Precision-Recall curves unavailable: no positive support for selected classes."
        )

    _handle_downloads(
        summary=summary,
        optimal_df=optimal_thresholds_df,
        predictions_df=predictions_df,
        thresholds_map=per_class_thresholds,
        classes=classes,
        confusion_fig=confusion_fig,
        confusion_fig_norm=confusion_fig_norm,
        roc_figures=roc_figures,
    )


if __name__ == "__main__":
    main()
