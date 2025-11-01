"""
Activation utilities for transforming model score matrices prior to thresholding.

These helpers are intentionally vectorised to support large (50k+) sample batches
without Python loops.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

ActivationType = Literal["none", "softmax", "sigmoid", "sigmoid_5"]


def _as_float_array(values: np.ndarray | list | tuple) -> np.ndarray:
    """Convert arbitary array-like values to a float64 NumPy array."""
    return np.asarray(values, dtype=np.float64)


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax across the class axis.

    Parameters
    ----------
    logits:
        Raw score matrix shaped (..., n_classes).
    axis:
        Axis across which softmax is applied (defaults to last axis).

    Returns
    -------
    np.ndarray
        Probability matrix with the same shape as ``logits``.
    """
    scores = _as_float_array(logits)

    if scores.ndim == 0:
        return np.array(1.0, dtype=np.float64)

    max_scores = np.max(scores, axis=axis, keepdims=True)
    stabilised = scores - max_scores
    np.nan_to_num(stabilised, copy=False)

    exp_scores = np.exp(stabilised)
    sum_exp = np.sum(exp_scores, axis=axis, keepdims=True)
    sum_exp[sum_exp == 0.0] = 1.0  # prevent division by zero

    probabilities = exp_scores / sum_exp
    return probabilities


def sigmoid(x: np.ndarray | list | tuple) -> np.ndarray:
    """
    Compute the logistic sigmoid elementwise in a numerically stable manner.
    """
    z = _as_float_array(x)
    result = np.empty_like(z, dtype=np.float64)

    positive_mask = z >= 0
    negative_mask = ~positive_mask

    result[positive_mask] = 1.0 / (1.0 + np.exp(-z[positive_mask]))

    exp_z = np.exp(z[negative_mask])
    result[negative_mask] = exp_z / (1.0 + exp_z)

    return result


def sigmoid_5(x: np.ndarray | list | tuple) -> np.ndarray:
    """
    Custom sigmoid variant with a scale factor of 5:
        1 / (1 + exp(-x / 5))
    """
    scaled = _as_float_array(x) / 5.0
    return sigmoid(scaled)


def apply_activation(
    scores: np.ndarray | list | tuple,
    activation_type: str | None,
) -> np.ndarray:
    """
    Apply the requested activation function to the provided scores.

    Parameters
    ----------
    scores:
        Score matrix (n_samples, n_classes).
    activation_type:
        One of {"none", "softmax", "sigmoid", "sigmoid_5"} (case-insensitive).
        ``None`` defaults to "none".

    Returns
    -------
    np.ndarray
        Activated score matrix.
    """
    activation = (activation_type or "none").lower()

    if activation in {"none", "raw"}:
        return _as_float_array(scores)
    if activation == "softmax":
        return softmax(scores)
    if activation == "sigmoid":
        return sigmoid(scores)
    if activation in {"sigmoid_5", "sigmoid5"}:
        return sigmoid_5(scores)

    raise ValueError(
        f"Unsupported activation_type '{activation_type}'. "
        "Expected one of {'none', 'softmax', 'sigmoid', 'sigmoid_5'}."
    )


__all__ = [
    "ActivationType",
    "softmax",
    "sigmoid",
    "sigmoid_5",
    "apply_activation",
]
