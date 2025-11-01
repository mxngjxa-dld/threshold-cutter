"""
Export utilities for persisting model artefacts and evaluation results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
from matplotlib.figure import Figure


import pandas as pd

FIGURE_DIR = Path("app") / "outputs" / "figures"
PREDICTION_DIR = Path("app") / "outputs" / "predictions"
REPORT_DIR = Path("app") / "outputs" / "reports"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _snake_case(name: str) -> str:
    safe = "".join(char if char.isalnum() else "_" for char in name.lower())
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_")


def generate_timestamp() -> str:
    """
    Generate a consistent timestamp format for filenames.
    """
    return pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")


def save_classification_report(
    report: dict,
    timestamp: str | None = None,
    base_name: str = "classification_report",
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """
    Persist a classification report dictionary as JSON and CSV.
    """
    timestamp = timestamp or generate_timestamp()
    output_dir = _ensure_dir(output_dir or REPORT_DIR)

    base = output_dir / f"{_snake_case(base_name)}_{timestamp}"

    json_path = base.with_suffix(".json")
    csv_path = base.with_suffix(".csv")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    df = pd.DataFrame(report)
    df.to_csv(csv_path, index=False)

    return {"json": json_path, "csv": csv_path}


def save_confusion_matrix_image(
    fig,
    timestamp: str | None = None,
    base_name: str = "confusion_matrix",
    output_dir: Path | None = None,
    dpi: int = 300,
) -> Path:
    """
    Save the confusion matrix Matplotlib figure as a PNG file.
    """
    timestamp = timestamp or generate_timestamp()
    output_dir = _ensure_dir(output_dir or FIGURE_DIR)
    path = output_dir / f"{_snake_case(base_name)}_{timestamp}.png"

    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def save_roc_images(
    figures: dict[str, object],
    classes: Iterable[str],
    timestamp: str | None = None,
    output_dir: Path | None = None,
    dpi: int = 300,
    base_prefix: str = "roc_curve",
) -> dict[str, Path]:
    """
    Save a dictionary of ROC Matplotlib figures returned by ``plot_per_class_roc``.
    """
    timestamp = timestamp or generate_timestamp()
    output_dir = _ensure_dir(output_dir or FIGURE_DIR)

    saved_paths: dict[str, Path] = {}
    for cls in classes:
        fig = figures.get(cls)
        if fig is None:
            continue
        if isinstance(fig, Figure):  # Proper type guard for matplotlib Figure
            filename = f"{_snake_case(base_prefix)}_{_snake_case(cls)}_{timestamp}.png"
            path = output_dir / filename
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            saved_paths[cls] = path

    combined_fig = figures.get("combined")
    if combined_fig is not None and isinstance(combined_fig, Figure):
        filename = f"{_snake_case(base_prefix)}_combined_{timestamp}.png"
        path = output_dir / filename
        combined_fig.savefig(path, dpi=dpi, bbox_inches="tight")
        saved_paths["combined"] = path

    return saved_paths


def save_predictions_csv(
    df: pd.DataFrame,
    thresholds: dict[str, float] | pd.DataFrame,
    timestamp: str | None = None,
    base_name: str = "predictions",
    output_dir: Path | None = None,
) -> Path:
    """
    Save predictions with associated scores and thresholds.
    """
    timestamp = timestamp or generate_timestamp()
    output_dir = _ensure_dir(output_dir or PREDICTION_DIR)

    path = output_dir / f"{_snake_case(base_name)}_{timestamp}.csv"
    df_copy = df.copy()

    if isinstance(thresholds, pd.DataFrame):
        threshold_series = thresholds.squeeze()
    else:
        threshold_series = pd.Series(thresholds, name="threshold")

    threshold_df = threshold_series.reset_index()
    threshold_df.columns = ["class", "threshold"]

    # join thresholds into dataframe (one row per prediction)
    if "predicted_class" in df_copy.columns:
        df_copy = df_copy.merge(
            threshold_df,
            how="left",
            left_on="predicted_class",
            right_on="class",
        )
        df_copy.drop(columns=["class"], inplace=True)

    df_copy.to_csv(path, index=False)
    return path


__all__ = [
    "generate_timestamp",
    "save_classification_report",
    "save_confusion_matrix_image",
    "save_roc_images",
    "save_predictions_csv",
]
