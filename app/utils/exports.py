"""
Export utilities for persisting model artefacts and evaluation results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from matplotlib.figure import Figure

import numpy as np
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


_CLASS_LABEL_KEYS: tuple[str, ...] = ("class", "index", "label", "name")


def _coerce_class_label(entry: Mapping[str, Any]) -> str:
    for key in _CLASS_LABEL_KEYS:
        if key in entry:
            value = entry[key]
            if value is None:
                return ""
            return str(value)
    return ""


def _is_non_string_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _flatten_report_for_csv(report: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for section, payload in report.items():
        section_name = str(section)
        if isinstance(payload, Mapping):
            for metric, value in payload.items():
                rows.append(
                    {
                        "section": section_name,
                        "class": "",
                        "metric": str(metric),
                        "value": value,
                    }
                )
            continue

        if _is_non_string_sequence(payload):
            for entry in payload:
                if isinstance(entry, Mapping):
                    class_label = _coerce_class_label(entry)
                    for metric, value in entry.items():
                        if metric in _CLASS_LABEL_KEYS:
                            continue
                        rows.append(
                            {
                                "section": section_name,
                                "class": class_label,
                                "metric": str(metric),
                                "value": value,
                            }
                        )
                else:
                    rows.append(
                        {
                            "section": section_name,
                            "class": "",
                            "metric": "",
                            "value": entry,
                        }
                    )
            continue

        rows.append(
            {
                "section": section_name,
                "class": "",
                "metric": "",
                "value": payload,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["section", "class", "metric", "value"])

    df = pd.DataFrame(rows, columns=["section", "class", "metric", "value"])
    df["class"] = df["class"].fillna("").astype(str)
    df["metric"] = df["metric"].fillna("").astype(str)
    return df


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def generate_timestamp() -> str:
    """
    Generate a consistent timestamp format for filenames.
    """
    return pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")


def save_classification_report(
    report: Mapping[str, Any],
    timestamp: str | None = None,
    base_name: str = "classification_report",
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """
    Persist a classification report dictionary as JSON and CSV.

    The JSON file mirrors the provided nested structure. The CSV output is flattened into
    columns (section, class, metric, value) to avoid pandas constructor ambiguities with
    mixed mappings and sequences.
    """
    timestamp = timestamp or generate_timestamp()
    output_dir = _ensure_dir(output_dir or REPORT_DIR)

    base = output_dir / f"{_snake_case(base_name)}_{timestamp}"

    json_path = base.with_suffix(".json")
    csv_path = base.with_suffix(".csv")

    report_dict = dict(report)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False, cls=NpEncoder)

    df = _flatten_report_for_csv(report_dict)
    df.to_csv(csv_path, index=False)

    return {"json": json_path, "csv": csv_path}


def save_optimal_metrics(
    optimal_df: pd.DataFrame,
    timestamp: str | None = None,
    base_name: str = "optimal_metrics",
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """
    Persist optimal threshold statistics (e.g., Youden's J) to JSON and CSV artefacts.
    """
    timestamp = timestamp or generate_timestamp()
    output_dir = _ensure_dir(output_dir or REPORT_DIR)

    base = output_dir / f"{_snake_case(base_name)}_{timestamp}"
    json_path = base.with_suffix(".json")
    csv_path = base.with_suffix(".csv")

    serialisable = optimal_df.reset_index()
    if "index" in serialisable.columns and "class" not in serialisable.columns:
        serialisable = serialisable.rename(columns={"index": "class"})

    serialisable.to_csv(csv_path, index=False)

    records = serialisable.to_dict(orient="records")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False, cls=NpEncoder)

    return {"json": json_path, "csv": csv_path}


def save_confusion_matrix_image(
    fig,
    norm: bool, 
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
    type = "norm" if norm else "raw"
    path = output_dir / f"{_snake_case(base_name)}_{timestamp}_{type}.png"

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
    "save_optimal_metrics",
    "save_confusion_matrix_image",
    "save_roc_images",
    "save_predictions_csv",
]
