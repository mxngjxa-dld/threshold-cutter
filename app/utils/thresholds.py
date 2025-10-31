"""
Thresholding utilities for transforming activated class scores into predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .activations import ActivationType, apply_activation


@dataclass(frozen=True)
class ThresholdResult:
    """Container returned by ``predict_with_thresholds``."""

    predictions: np.ndarray
    activated_scores: np.ndarray
    mask: np.ndarray
    masked_scores: np.ndarray


def _ensure_2d_scores(scores: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(scores, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(
            "Score matrix must be 2-dimensional with shape (n_samples, n_classes)."
        )
    return array


def _normalise_classes(classes: Iterable) -> tuple:
    classes_tuple = tuple(classes)
    if not classes_tuple:
        raise ValueError("Classes iterable cannot be empty.")
    return classes_tuple


def _build_threshold_array(
    thresholds: Mapping[str, float] | Sequence[float] | float | np.ndarray | None,
    classes: Sequence[str],
    default: float = 0.0,
) -> np.ndarray:
    classes = list(classes)

    if thresholds is None:
        return np.full((1, len(classes)), float(default), dtype=np.float64)

    if isinstance(thresholds, Mapping):
        return np.array(
            [[float(thresholds.get(cls, default)) for cls in classes]],
            dtype=np.float64,
        )

    if np.isscalar(thresholds):
        return np.full((1, len(classes)), float(thresholds), dtype=np.float64)

    threshold_arr = np.asarray(thresholds, dtype=np.float64)
    if threshold_arr.ndim == 1 and threshold_arr.shape[0] == len(classes):
        return threshold_arr[np.newaxis, :]

    if threshold_arr.ndim == 2:
        if threshold_arr.shape == (1, len(classes)):
            return threshold_arr
        if threshold_arr.shape == (len(classes), 1):
            return threshold_arr.T
        if threshold_arr.shape == (len(classes), len(classes)):
            return np.diag(threshold_arr)[np.newaxis, :]

    raise ValueError(
        "Thresholds must be a scalar, mapping, or iterable compatible with class length."
    )


def predict_with_thresholds(
    scores: np.ndarray | Sequence[Sequence[float]],
    classes: Iterable[str],
    thresholds: Mapping[str, float] | Sequence[float] | float | None = None,
    activation: ActivationType | str | None = "none",
    fallback_class: str | None = None,
) -> ThresholdResult:
    """
    Apply activation + thresholding and obtain final class predictions.

    Parameters
    ----------
    scores:
        Raw score matrix shaped (n_samples, n_classes).
    classes:
        Iterable of class labels corresponding to the score columns.
    thresholds:
        Scalar, iterable, or mapping providing per-class thresholds.
    activation:
        Activation function to apply before thresholding.
    fallback_class:
        Class used when no thresholds are exceeded. Defaults to the first class.

    Returns
    -------
    ThresholdResult
        Dataclass containing predictions, activated scores, threshold mask, and masked scores.
    """
    class_labels = _normalise_classes(classes)
    raw_scores = _ensure_2d_scores(scores)

    activated = apply_activation(raw_scores, activation)
    threshold_array = _build_threshold_array(thresholds, class_labels, default=0.0)
    mask = apply_thresholds(activated, threshold_array, class_labels)
    masked_scores = np.where(mask, activated, -np.inf)

    predictions = select_predicted_class(
    masked_scores, fallback_class=fallback_class, classes=class_labels
    )

    return ThresholdResult(
        predictions=predictions,
        activated_scores=activated,
        mask=mask,
        masked_scores=masked_scores,
    )


def compute_optimal_thresholds_youden(
    y_true: Sequence,
    y_scores: np.ndarray | Sequence[Sequence[float]],
    classes: Iterable[str],
) -> pd.DataFrame:
    """
    Compute per-class thresholds maximising Youden's J statistic (TPR - FPR).
    """
    y_true_series = pd.Series(list(y_true))
    scores = _ensure_2d_scores(y_scores)
    class_labels = _normalise_classes(classes)

    if scores.shape[0] != len(y_true_series):
        raise ValueError(
            "y_scores and y_true must have the same number of rows/samples."
        )
    if scores.shape[1] != len(class_labels):
        raise ValueError(
            "y_scores column count must match the length of the classes iterable."
        )

    results = []
    for idx, cls in enumerate(class_labels):
        binary_true = (y_true_series == cls).astype(int)

        positives = int(binary_true.sum())
        negatives = len(binary_true) - positives

        if positives == 0 or negatives == 0:
            results.append(
                {
                    "class": cls,
                    "optimal_threshold": np.nan,
                    "youden_j": np.nan,
                    "tpr": 0.0,
                    "fpr": 1.0 if positives == 0 else 0.0,
                    "support_pos": positives,
                    "support_neg": negatives,
                }
            )
            continue

        fpr, tpr, thresholds = roc_curve(binary_true, scores[:, idx])
        youden_j = tpr - fpr
        best_idx = np.nanargmax(youden_j)
        results.append(
            {
                "class": cls,
                "optimal_threshold": float(thresholds[best_idx]),
                "youden_j": float(youden_j[best_idx]),
                "tpr": float(tpr[best_idx]),
                "fpr": float(fpr[best_idx]),
                "support_pos": positives,
                "support_neg": negatives,
            }
        )

    return pd.DataFrame(results).set_index("class")


def apply_thresholds(
    scores: np.ndarray | Sequence[Sequence[float]],
    thresholds: Mapping[str, float] | Sequence[float] | float | np.ndarray,
    classes: Iterable[str],
) -> np.ndarray:
    """
    Produce a boolean mask where scores meet/exceed the specified thresholds.
    """
    class_labels = _normalise_classes(classes)
    activated = _ensure_2d_scores(scores)
    threshold_array = _build_threshold_array(thresholds, class_labels, default=0.0)

    return activated >= threshold_array


def select_predicted_class(
    masked_scores: np.ndarray | pd.DataFrame,
    fallback_class: str | None,
    classes: Iterable[str] | None = None,
) -> np.ndarray:
    """
    Choose the predicted class from masked scores (values below threshold set to -inf).
    """
    if isinstance(masked_scores, pd.DataFrame):
        scores_array = masked_scores.to_numpy(dtype=np.float64, copy=False)
        inferred_classes = tuple(str(col) for col in masked_scores.columns)
    else:
        scores_array = np.asarray(masked_scores, dtype=np.float64)
        inferred_classes = tuple(classes) if classes is not None else None

    if scores_array.ndim != 2:
        raise ValueError("masked_scores must be a 2D array-like.")
    if inferred_classes is None or len(inferred_classes) != scores_array.shape[1]:
        raise ValueError(
            "Class labels must be provided when masked_scores lacks column metadata."
        )

    if fallback_class is None:
        fallback_class = inferred_classes[0]

    best_indices = np.argmax(scores_array, axis=1)
    has_valid = np.isfinite(scores_array).any(axis=1)

    predictions = np.array(
        [inferred_classes[idx] for idx in best_indices], dtype=object
    )
    predictions[~has_valid] = fallback_class
    return predictions


__all__ = [
    "ThresholdResult",
    "apply_thresholds",
    "compute_optimal_thresholds_youden",
    "predict_with_thresholds",
    "select_predicted_class",
]