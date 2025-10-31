"""
Data ingestion and validation helpers.

Supports both "wide" (one row per sample with ``logit_{class}`` columns) and
"long" (multiple rows per sample with ``predicted_category``/``logit_score``)
table layouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping

import numpy as np
import pandas as pd

ROW_INDEX_COLUMN = "__row_id__"
WIDE_PREFIXES = ("logit_", "score_", "prob_", "p_")
CLASS_COLUMN_CANDIDATES = (
    "class",
    "label",
    "category",
    "predicted_category",
    "target_class",
)
SCORE_COLUMN_CANDIDATES = (
    "logit_score",
    "score",
    "probability",
    "value",
    "confidence",
)
SAMPLE_ID_COLUMN_CANDIDATES = (
    "sample_id",
    "id",
    "row_id",
    "record_id",
    "observation_id",
)


@dataclass(frozen=True)
class DataMetadata:
    """Descriptor returned by ``validate_data``."""

    format: Literal["wide", "long"]
    classes: tuple[str, ...]
    class_to_column: Mapping[str, str]
    long_class_column: str | None = None
    long_score_column: str | None = None
    sample_id_column: str | None = None


def load_csv(
    file_path: str | Path,
    delimiter: str | None = None,
    encoding: str = "utf-8",
    **read_csv_kwargs,
) -> pd.DataFrame:
    """
    Load a CSV file with optional delimiter detection.

    Parameters
    ----------
    file_path:
        Path to the CSV file.
    delimiter:
        Optional delimiter override. If omitted the function attempts to sniff
        the delimiter from the first non-empty line.
    encoding:
        File encoding passed to :func:`pandas.read_csv`.
    read_csv_kwargs:
        Extra keyword arguments forwarded to :func:`pandas.read_csv`.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    sep = delimiter or _sniff_delimiter(path)
    df = pd.read_csv(path, sep=sep, encoding=encoding, **read_csv_kwargs)
    if df.empty:
        raise ValueError("Loaded CSV is empty; please provide a dataset with rows.")
    return df


def _sniff_delimiter(path: Path) -> str:
    sample_bytes = path.read_bytes()
    first_chunk = sample_bytes.splitlines()
    for line in first_chunk:
        if not line:
            continue
        decoded = line.decode("utf-8", errors="ignore")
        for candidate in [",", "\t", ";", "|"]:
            if candidate in decoded:
                return candidate
    return ","


def validate_data(df: pd.DataFrame) -> DataMetadata:
    """
    Validate a dataframe and infer metadata needed for downstream processing.
    """
    if "true_label" not in df.columns:
        raise ValueError(
            "Input data must contain a 'true_label' column representing ground truth."
        )

    column_lookup = {col.lower(): col for col in df.columns}

    wide_map: dict[str, str] = {}
    for column in df.columns:
        lower = column.lower()
        for prefix in WIDE_PREFIXES:
            if lower.startswith(prefix):
                class_name = column[len(prefix) :].strip()
                if class_name:
                    wide_map[class_name] = column
                break

    if wide_map:
        classes = tuple(wide_map.keys())
        _validate_numeric_columns(df, wide_map.values())
        return DataMetadata(
            format="wide",
            classes=classes,
            class_to_column=wide_map,
        )

    class_column = _pick_first_available(column_lookup, CLASS_COLUMN_CANDIDATES)
    score_column = _pick_first_available(column_lookup, SCORE_COLUMN_CANDIDATES)
    if not class_column or not score_column:
        raise ValueError(
            "Unable to infer class/score columns for long-format data. "
            "Expected columns such as 'predicted_category' and 'logit_score'."
        )

    sample_id_column = _pick_first_available(column_lookup, SAMPLE_ID_COLUMN_CANDIDATES)
    _validate_numeric_columns(df, [score_column])

    classes = tuple(df[class_column].astype("string", copy=False).dropna().unique())
    if not classes:
        raise ValueError(
            "No class labels detected in column "
            f"'{class_column}'. Ensure the column contains non-null values."
        )

    return DataMetadata(
        format="long",
        classes=classes,
        class_to_column={},
        long_class_column=class_column,
        long_score_column=score_column,
        sample_id_column=sample_id_column,
    )


def _validate_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise TypeError(
                f"Column '{column}' must contain numeric scores for activation/thresholding."
            )


def _pick_first_available(
    column_lookup: Mapping[str, str], candidates: Iterable[str]
) -> str | None:
    for candidate in candidates:
        if candidate in column_lookup:
            return column_lookup[candidate]
    return None


def prepare_score_matrix(
    df: pd.DataFrame,
    classes: Iterable[str] | None = None,
    metadata: DataMetadata | None = None,
    return_index: bool = False,
) -> np.ndarray | tuple[np.ndarray, pd.Index]:
    """
    Convert the dataframe into a dense score matrix of shape (n_samples, n_classes).

    Parameters
    ----------
    df:
        Source dataframe containing score columns.
    classes:
        Optional explicit class ordering.
    metadata:
        Pre-computed metadata from :func:`validate_data`.
    return_index:
        When True, also return the row index used to align the resulting matrix.
    """
    metadata = metadata or validate_data(df)
    classes_tuple = tuple(classes) if classes else metadata.classes

    index_values: pd.Index | None = None

    if metadata.format == "wide":
        ordered_columns = [metadata.class_to_column[cls] for cls in classes_tuple]
        scores = df.loc[:, ordered_columns].to_numpy(dtype=np.float64, copy=False)
        index_values = df.index
    else:
        class_column = metadata.long_class_column
        score_column = metadata.long_score_column
        if class_column is None or score_column is None:
            raise RuntimeError("Long-format metadata missing required column references.")

        working_df = df.copy()
        index_column = metadata.sample_id_column
        if index_column is None:
            working_df[ROW_INDEX_COLUMN] = np.arange(len(working_df), dtype=np.int64)
            index_column = ROW_INDEX_COLUMN

        pivot = (
            working_df.pivot_table(
                index=index_column,
                columns=class_column,
                values=score_column,
                aggfunc="first",
            )
            .reindex(columns=classes_tuple)
            .sort_index()
        )

        pivot = pivot.astype(np.float64)
        pivot = pivot.fillna(np.nan)
        scores = pivot.to_numpy(dtype=np.float64, copy=False)
        index_values = pivot.index

    processed = handle_missing_scores(scores)
    if return_index:
        assert index_values is not None
        return processed, index_values
    return processed


def handle_missing_scores(scores: np.ndarray | pd.DataFrame) -> np.ndarray:
    """
    Replace NaN scores with -inf to ensure they are ignored post-activation.
    """
    array = np.asarray(scores, dtype=np.float64)
    np.nan_to_num(array, nan=-np.inf, copy=False)
    return array


_majority_cache: dict[tuple, object] = {}


def get_majority_class(y_true: Iterable) -> object:
    """
    Compute and cache the majority (mode) class label.

    Uses a lightweight dictionary cache keyed by a tuple representation of the
    labels. This avoids repeated passes over large datasets when threshold
    updates request the fallback class frequently.
    """
    values = tuple(pd.Series(y_true).tolist())
    if not values:
        raise ValueError("Cannot compute majority class from empty labels.")
    cached = _majority_cache.get(values)
    if cached is not None:
        return cached

    counts = pd.Series(values).value_counts()
    majority = counts.idxmax()
    _majority_cache[values] = majority
    return majority


__all__ = [
    "DataMetadata",
    "handle_missing_scores",
    "load_csv",
    "prepare_score_matrix",
    "validate_data",
    "get_majority_class",
]