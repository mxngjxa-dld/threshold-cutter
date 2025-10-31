import numpy as np
import pandas as pd
import pytest

from app.utils.thresholds import (
    compute_optimal_thresholds_youden,
    predict_with_thresholds,
    select_predicted_class,
)


@pytest.fixture
def sample_scores():
    classes = ["cat", "dog", "rabbit"]
    scores = np.array(
        [
            [0.9, 0.2, -0.5],
            [0.1, 0.6, 0.3],
            [0.2, 0.2, 0.1],
            [-0.2, 0.8, 0.4],
        ]
    )
    y_true = np.array(["cat", "dog", "cat", "dog"])
    return scores, y_true, classes


def test_predict_with_thresholds_basic(sample_scores):
    scores, y_true, classes = sample_scores
    thresholds = {"cat": 0.3, "dog": 0.5, "rabbit": 0.4}
    result = predict_with_thresholds(
        scores,
        classes,
        thresholds=thresholds,
        activation="sigmoid",
        fallback_class="cat",
    )

    assert result.activated_scores.shape == scores.shape
    assert result.mask.shape == scores.shape
    assert result.masked_scores.shape == scores.shape
    assert len(result.predictions) == scores.shape[0]
    assert set(result.predictions).issubset(set(classes + ["cat"]))


def test_select_predicted_class_with_no_valid_scores():
    masked = np.full((3, 2), -np.inf)
    predictions = select_predicted_class(masked, fallback_class="fallback", classes=["a", "b"])
    assert np.all(predictions == "fallback")


def test_compute_optimal_thresholds_handles_missing_support(sample_scores):
    scores, y_true, classes = sample_scores
    y_modified = y_true.copy()
    y_modified[y_modified == "cat"] = "dog"
    df = compute_optimal_thresholds_youden(
        y_true=y_modified,
        y_scores=scores,
        classes=classes,
    )
    assert set(df.index) == set(classes)
    assert pd.isna(df.loc["rabbit", "optimal_threshold"])
    assert df.loc["dog", "support_pos"] > 0
    assert df.loc["cat", "support_pos"] == 0


def test_predict_with_thresholds_softmax_scalar_threshold(sample_scores):
    scores, y_true, classes = sample_scores
    result = predict_with_thresholds(
        scores,
        classes,
        thresholds=0.5,
        activation="softmax",
        fallback_class="dog",
    )
    assert result.mask.dtype == bool
    assert len(result.predictions) == scores.shape[0]
    assert "dog" in result.predictions