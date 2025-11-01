"""
Metric computation helpers for the multiclass threshold tuner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

ArrayLike = Iterable


@dataclass(frozen=True)
class MetricSummary:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: pd.DataFrame
    micro_auc: float | None
    macro_auc: float | None
    youden_by_class: pd.DataFrame


def _as_numpy_labels(values: ArrayLike) -> np.ndarray:
    array = np.asarray(list(values))
    if array.size == 0:
        raise ValueError("Metric computations require non-empty label arrays.")
    return array


def compute_classification_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> dict[str, float]:
    """
    Return a dictionary of aggregate classification metrics.
    """
    y_true_arr = _as_numpy_labels(y_true)
    y_pred_arr = _as_numpy_labels(y_pred)

    labels = np.unique(np.concatenate([y_true_arr, y_pred_arr]))
    accuracy = float(accuracy_score(y_true_arr, y_pred_arr))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=labels,
        average=None,
        zero_division=0.0,
    )
    macro_precision = float(np.nanmean(precision))
    macro_recall = float(np.nanmean(recall))
    macro_f1 = float(np.nanmean(f1))

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def per_class_roc_and_j(
    y_true: ArrayLike,
    y_scores: np.ndarray,
    classes: Iterable[str],
) -> pd.DataFrame:
    """
    Compute ROC curve data, AUC, and Youden's J statistic per class.
    """
    y_true_arr = _as_numpy_labels(y_true)
    scores = np.asarray(y_scores, dtype=np.float64)
    class_labels = list(classes)

    if scores.ndim != 2:
        raise ValueError("y_scores must be a 2D array of shape (n_samples, n_classes).")
    if scores.shape[0] != len(y_true_arr):
        raise ValueError("Number of score rows must match number of labels.")
    if scores.shape[1] != len(class_labels):
        raise ValueError(
            "Number of score columns must match the length of the classes iterable."
        )

    records = []
    for idx, cls in enumerate(class_labels):
        binary_true = (y_true_arr == cls).astype(int)

        positives = int(binary_true.sum())
        negatives = len(binary_true) - positives
        if positives == 0 or negatives == 0:
            records.append(
                {
                    "class": cls,
                    "auc": np.nan,
                    "youden_j": np.nan,
                    "optimal_threshold": np.nan,
                    "tpr": 0.0,
                    "fpr": 1.0 if positives == 0 else 0.0,
                    "roc_curve": (np.array([0.0, 1.0]), np.array([0.0, 1.0])),
                    "support_pos": positives,
                    "support_neg": negatives,
                }
            )
            continue

        if positives == 1 or negatives == 1:
            records.append(
                {
                    "class": cls,
                    "auc": np.nan,
                    "youden_j": np.nan,
                    "optimal_threshold": np.nan,
                    "tpr": np.nan,
                    "fpr": np.nan,
                    "roc_curve": (np.array([0.0, 1.0]), np.array([0.0, 1.0])),
                    "support_pos": positives,
                    "support_neg": negatives,
                }
            )
            continue

        fpr, tpr, thresholds = roc_curve(binary_true, scores[:, idx])
        youden = tpr - fpr
        best_idx = np.nanargmax(youden)

        class_auc = float(
            auc(fpr, tpr)
            if np.isfinite(fpr).all() and np.isfinite(tpr).all()
            else np.nan
        )

        records.append(
            {
                "class": cls,
                "auc": class_auc,
                "youden_j": float(youden[best_idx]),
                "optimal_threshold": float(thresholds[best_idx]),
                "tpr": float(tpr[best_idx]),
                "fpr": float(fpr[best_idx]),
                "roc_curve": (fpr, tpr),
                "support_pos": int(positives),
                "support_neg": int(negatives),
            }
        )

    df = pd.DataFrame(records).set_index("class")
    return df


def compute_micro_macro_auc(
    y_true: ArrayLike,
    y_scores: np.ndarray,
    classes: Iterable[str],
) -> tuple[float | None, float | None]:
    """
    Compute micro and macro averaged AUC scores.
    """
    y_true_arr = _as_numpy_labels(y_true)
    scores = np.asarray(y_scores, dtype=np.float64)
    class_labels = list(classes)

    if scores.shape[0] != len(y_true_arr):
        raise ValueError("y_scores rows must equal number of labels.")
    if scores.shape[1] != len(class_labels):
        raise ValueError("y_scores columns must match class count.")

    # convert to one-hot for ROC AUC; gracefully handle missing classes
    one_hot = pd.get_dummies(y_true_arr, drop_first=False).reindex(
        columns=class_labels, fill_value=0
    )
    if one_hot.values.ndim != 2:
        raise ValueError("Failed to construct one-hot encoded labels.")

    try:
        micro_auc = float(
            roc_auc_score(one_hot.values, scores, average="micro", multi_class="ovr")
        )
    except ValueError:
        micro_auc = None

    try:
        macro_auc = float(
            roc_auc_score(one_hot.values, scores, average="macro", multi_class="ovr")
        )
    except ValueError:
        macro_auc = None

    return micro_auc, macro_auc


def create_metrics_summary(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_scores: np.ndarray,
    classes: Iterable[str],
) -> MetricSummary:
    """
    Aggregate metrics into a structured summary object.
    """
    agg_metrics = compute_classification_metrics(y_true, y_pred)
    per_class_df = per_class_roc_and_j(y_true, y_scores, classes)
    micro_auc, macro_auc = compute_micro_macro_auc(y_true, y_scores, classes)

    summary_df = per_class_df.copy()
    summary_df["support_total"] = summary_df["support_pos"] + summary_df["support_neg"]
    summary_df["auc"] = summary_df["auc"].astype(float)

    return MetricSummary(
        accuracy=agg_metrics["accuracy"],
        macro_precision=agg_metrics["macro_precision"],
        macro_recall=agg_metrics["macro_recall"],
        macro_f1=agg_metrics["macro_f1"],
        per_class=summary_df,
        micro_auc=micro_auc,
        macro_auc=macro_auc,
        youden_by_class=per_class_df[["youden_j", "optimal_threshold", "tpr", "fpr"]],
    )


__all__ = [
    "MetricSummary",
    "compute_classification_metrics",
    "per_class_roc_and_j",
    "compute_micro_macro_auc",
    "create_metrics_summary",
]
