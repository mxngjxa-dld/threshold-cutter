from __future__ import annotations

import json

import pandas as pd
import pytest

from app.utils.exports import save_classification_report, save_optimal_metrics


def test_save_classification_report_flatten(tmp_path):
    report = {
        "overall": {
            "accuracy": 0.91,
            "macro_f1": 0.88,
        },
        "per_class": [
            {"class": "cat", "precision": 0.9, "recall": 0.85},
            {"class": "dog", "precision": 0.87, "recall": 0.8},
        ],
        "youden": [
            {"class": "cat", "optimal_threshold": 0.42, "youden_j": 0.55},
            {"class": "dog", "optimal_threshold": 0.37, "youden_j": 0.49},
        ],
    }

    paths = save_classification_report(
        report,
        timestamp="20250101_000000",
        output_dir=tmp_path,
    )

    json_path = paths["json"]
    csv_path = paths["csv"]

    assert json_path.exists()
    assert csv_path.exists()

    with json_path.open("r", encoding="utf-8") as f:
        saved_report = json.load(f)
    assert saved_report == report

    df = pd.read_csv(csv_path)
    assert list(df.columns) == ["section", "class", "metric", "value"]

    overall_accuracy = df[
        (df["section"] == "overall") & (df["metric"] == "accuracy")
    ]
    assert not overall_accuracy.empty
    assert overall_accuracy["value"].iloc[0] == pytest.approx(0.91)

    per_class_precision = df[
        (df["section"] == "per_class")
        & (df["class"] == "cat")
        & (df["metric"] == "precision")
    ]
    assert not per_class_precision.empty
    assert per_class_precision["value"].iloc[0] == pytest.approx(0.9)

    youden_threshold = df[
        (df["section"] == "youden")
        & (df["class"] == "dog")
        & (df["metric"] == "optimal_threshold")
    ]
    assert not youden_threshold.empty
    assert youden_threshold["value"].iloc[0] == pytest.approx(0.37)


def test_save_optimal_metrics_outputs(tmp_path):
    optimal_df = pd.DataFrame(
        {
            "optimal_threshold": [0.42, 0.37],
            "youden_j": [0.55, 0.49],
            "tpr": [0.8, 0.75],
            "fpr": [0.2, 0.25],
        },
        index=pd.Index(["cat", "dog"], name="class"),
    )

    paths = save_optimal_metrics(
        optimal_df,
        timestamp="20250101_010101",
        output_dir=tmp_path,
    )

    json_path = paths["json"]
    csv_path = paths["csv"]

    assert json_path.exists()
    assert csv_path.exists()

    df_csv = pd.read_csv(csv_path)
    assert set(df_csv["class"]) == {"cat", "dog"}
    assert {"optimal_threshold", "youden_j", "tpr", "fpr"}.issubset(df_csv.columns)

    with json_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    assert isinstance(records, list)
    assert {entry["class"] for entry in records} == {"cat", "dog"}
    for entry in records:
        assert set(entry) >= {"class", "optimal_threshold", "youden_j"}