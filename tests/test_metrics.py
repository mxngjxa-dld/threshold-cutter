import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from app.utils.metrics import (
    MetricSummary,
    compute_classification_metrics,
    compute_micro_macro_auc,
    create_metrics_summary,
    per_class_roc_and_j,
)


@pytest.fixture
def sample_outputs():
    classes = ["a", "b", "c"]
    y_true = np.array(["a", "b", "c", "a", "b", "c"])
    y_pred = np.array(["a", "b", "a", "c", "b", "c"])
    scores = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.3, 0.2, 0.5],
            [0.2, 0.3, 0.5],
            [0.1, 0.7, 0.2],
            [0.2, 0.1, 0.7],
        ]
    )
    return y_true, y_pred, scores, classes


def test_compute_classification_metrics(sample_outputs):
    y_true, y_pred, _, _ = sample_outputs
    metrics = compute_classification_metrics(y_true, y_pred)
    assert set(metrics.keys()) == {
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    }
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_per_class_roc_and_j(sample_outputs):
    y_true, _, scores, classes = sample_outputs
    df = per_class_roc_and_j(y_true, scores, classes)
    assert set(df.columns) >= {
        "auc",
        "youden_j",
        "optimal_threshold",
        "tpr",
        "fpr",
        "roc_curve",
        "support_pos",
        "support_neg",
    }
    assert set(df.index) == set(classes)
    for cls in classes:
        curve = df.loc[cls, "roc_curve"]
        assert isinstance(curve, tuple)
        assert len(curve) == 2


def test_compute_micro_macro_auc(sample_outputs):
    y_true, _, scores, classes = sample_outputs
    micro_auc, macro_auc = compute_micro_macro_auc(y_true, scores, classes)
    manual_micro = roc_auc_score(
        pd.get_dummies(y_true, drop_first=False).values,
        scores,
        average="micro",
        multi_class="ovr",
    )
    manual_macro = roc_auc_score(
        pd.get_dummies(y_true, drop_first=False).values,
        scores,
        average="macro",
        multi_class="ovr",
    )
    assert micro_auc == pytest.approx(manual_micro)
    assert macro_auc == pytest.approx(manual_macro)


def test_create_metrics_summary(sample_outputs):
    y_true, y_pred, scores, classes = sample_outputs
    summary = create_metrics_summary(y_true, y_pred, scores, classes)
    assert isinstance(summary, MetricSummary)
    assert summary.per_class.index.tolist() == list(classes)
    assert summary.youden_by_class.index.tolist() == list(classes)

    support_totals = summary.per_class["support_total"]
    expected_supports = summary.per_class["support_pos"] + summary.per_class["support_neg"]
    pd.testing.assert_series_equal(support_totals, expected_supports, check_names=False)
    assert (support_totals == len(y_true)).all()


def test_per_class_roc_and_j_handles_missing_support():
    y_true = np.array(["x", "y", "y", "y"])
    scores = np.array(
        [
            [0.1, 0.9],
            [0.3, 0.7],
            [0.2, 0.8],
            [0.4, 0.6],
        ]
    )
    classes = ["x", "y"]
    df = per_class_roc_and_j(y_true, scores, classes)
    assert np.isnan(df.loc["x", "auc"])
    assert df.loc["y", "support_pos"] == 3
    assert df.loc["x", "support_pos"] == 1