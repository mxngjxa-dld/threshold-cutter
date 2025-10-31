"""
Data ingestion and validation helpers.

Supports both "wide" (one row per sample with ``logit_{class}`` columns) and
"long" (multiple rows per sample with ``predicted_category``/``logit_score``)
table layouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

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
TRUE_LABEL_COLUMN_CANDIDATES = (
    "true_label",
    "label",
    "actual",
    "ground_truth",
    "target",
    "response",
    "y_true",
)

SCORE_CLIP_ABS = 1e12
MISSING_SCORE_FILL_VALUE = -SCORE_CLIP_ABS


@dataclass(frozen=True)
class ColumnSelection:
    format: Literal["wide", "long"]
    true_label: str
    wide_score_columns: tuple[str, ...] = ()
    long_class_column: str | None = None
    long_score_column: str | None = None
    sample_id_column: str | None = None


@dataclass(frozen=True)
class ColumnCandidates:
    default_format: Literal["wide", "long"]
    true_label_options: tuple[str, ...]
    true_label_default: str
    long_class_options: tuple[str, ...]
    long_class_default: str | None
    long_score_options: tuple[str, ...]
    long_score_default: str | None
    sample_id_options: tuple[str, ...]
    sample_id_default: str | None
    wide_score_options: tuple[str, ...]
    wide_score_default: tuple[str, ...]


@dataclass(frozen=True)
class DataMetadata:
    """Descriptor returned by ``validate_data``."""

    format: Literal["wide", "long"]
    classes: tuple[str, ...]
    class_to_column: Mapping[str, str]
    true_label_column: str
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


def infer_column_candidates(df: pd.DataFrame) -> ColumnCandidates:
    """
    Inspect the dataframe and propose sensible default column mappings.
    """
    columns = list(df.columns)
    if not columns:
        raise ValueError("Column inference requires a dataframe with at least one column.")

    numeric_columns = list(df.select_dtypes(include=[np.number]).columns)
    string_like = list(df.select_dtypes(include=["object", "string", "category"]).columns)
    column_lookup = {col.lower(): col for col in columns}

    true_label_default = _pick_first_available(column_lookup, TRUE_LABEL_COLUMN_CANDIDATES)
    if true_label_default is None:
        true_label_default = string_like[0] if string_like else columns[0]
    true_label_options = tuple(columns)

    long_class_candidates = [col for col in string_like if col != true_label_default]
    if not long_class_candidates:
        long_class_candidates = [col for col in columns if col != true_label_default]
    long_class_lookup = {col.lower(): col for col in long_class_candidates}
    long_class_default = _pick_first_available(long_class_lookup, CLASS_COLUMN_CANDIDATES)
    if long_class_default is None and long_class_candidates:
        long_class_default = long_class_candidates[0]

    long_score_candidates = [col for col in numeric_columns if col != true_label_default]
    if not long_score_candidates:
        long_score_candidates = [col for col in columns if col != true_label_default]
    long_score_lookup = {col.lower(): col for col in long_score_candidates}
    long_score_default = _pick_first_available(long_score_lookup, SCORE_COLUMN_CANDIDATES)
    if long_score_default is None and long_score_candidates:
        long_score_default = long_score_candidates[0]

    sample_id_candidates = [col for col in columns if col != true_label_default]
    sample_id_string_like = [
        col for col in sample_id_candidates if col in string_like or pd.api.types.is_integer_dtype(df[col])
    ]
    if sample_id_string_like:
        sample_id_candidates = sample_id_string_like
    sample_id_lookup = {col.lower(): col for col in sample_id_candidates}
    sample_id_default = _pick_first_available(sample_id_lookup, SAMPLE_ID_COLUMN_CANDIDATES)

    detected_wide = list(_auto_detect_wide_columns(df, true_label_default))
    wide_score_default = tuple(dict.fromkeys(detected_wide))
    wide_score_candidates = [col for col in numeric_columns if col != true_label_default]
    if not wide_score_candidates:
        wide_score_candidates = [col for col in columns if col != true_label_default]
    wide_score_options = tuple(wide_score_candidates)

    default_format: Literal["wide", "long"] = "wide" if len(wide_score_default) >= 2 else "long"

    return ColumnCandidates(
        default_format=default_format,
        true_label_options=true_label_options,
        true_label_default=true_label_default,
        long_class_options=tuple(long_class_candidates),
        long_class_default=long_class_default,
        long_score_options=tuple(long_score_candidates),
        long_score_default=long_score_default,
        sample_id_options=tuple(sample_id_candidates),
        sample_id_default=sample_id_default,
        wide_score_options=wide_score_options,
        wide_score_default=wide_score_default,
    )


def validate_data(
    df: pd.DataFrame,
    column_selection: ColumnSelection | None = None,
) -> DataMetadata:
    """
    Validate a dataframe and infer metadata needed for downstream processing.
    """
    if df.empty:
        raise ValueError("Input data must contain at least one row.")

    column_lookup = {col.lower(): col for col in df.columns}

    if column_selection:
        true_label_column = column_selection.true_label
        if true_label_column not in df.columns:
            raise ValueError(
                f"Selected true label column '{true_label_column}' is not present in the dataset."
            )
    else:
        true_label_column = _pick_first_available(column_lookup, TRUE_LABEL_COLUMN_CANDIDATES)
        if true_label_column is None:
            string_candidates = list(df.select_dtypes(include=["object", "string", "category"]).columns)
            if not string_candidates:
                raise ValueError(
                    "Input data must contain a string-like true label column or you must specify one explicitly."
                )
            true_label_column = string_candidates[0]

    sample_id_column = (
        column_selection.sample_id_column
        if column_selection and column_selection.sample_id_column
        else _pick_first_available(column_lookup, SAMPLE_ID_COLUMN_CANDIDATES)
    )
    if sample_id_column and sample_id_column not in df.columns:
        raise ValueError(
            f"Selected sample identifier column '{sample_id_column}' is not present in the dataset."
        )
    if sample_id_column == true_label_column:
        sample_id_column = None

    force_format = column_selection.format if column_selection else None

    wide_columns: Sequence[str] = ()
    if force_format == "wide":
        wide_columns = column_selection.wide_score_columns or _auto_detect_wide_columns(df, true_label_column)
    elif force_format != "long":
        wide_columns = _auto_detect_wide_columns(df, true_label_column)

    if wide_columns:
        missing = [col for col in wide_columns if col not in df.columns]
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"Selected score columns not found in dataframe: {missing_str}")
        wide_map = _build_wide_map_from_columns(wide_columns)
        if not wide_map:
            raise ValueError("No class score columns detected for wide-format data.")
        _validate_numeric_columns(df, wide_map.values())
        classes = tuple(wide_map.keys())
        return DataMetadata(
            format="wide",
            classes=classes,
            class_to_column=wide_map,
            true_label_column=true_label_column,
            sample_id_column=sample_id_column,
        )

    if force_format == "wide":
        raise ValueError(
            "Wide-format data requires at least one numeric score column. "
            "Select the relevant columns or provide data in long format."
        )

    class_column = (
        column_selection.long_class_column
        if column_selection and column_selection.format == "long"
        else _pick_first_available(column_lookup, CLASS_COLUMN_CANDIDATES)
    )
    if class_column is None or class_column not in df.columns:
        raise ValueError(
            "Unable to infer class column for long-format data. "
            "Select a column containing predicted class labels."
        )

    score_column = (
        column_selection.long_score_column
        if column_selection and column_selection.format == "long"
        else _pick_first_available(column_lookup, SCORE_COLUMN_CANDIDATES)
    )
    if score_column is None or score_column not in df.columns:
        raise ValueError(
            "Unable to infer score column for long-format data. "
            "Select a numeric column containing class scores."
        )

    _validate_numeric_columns(df, [score_column])

    classes_series = df[class_column].astype("string", copy=False).dropna()
    classes = tuple(classes_series.unique())
    if not classes:
        raise ValueError(
            "No class labels detected in column "
            f"'{class_column}'. Ensure the column contains non-null values."
        )

    return DataMetadata(
        format="long",
        classes=classes,
        class_to_column={},
        true_label_column=true_label_column,
        long_class_column=class_column,
        long_score_column=score_column,
        sample_id_column=sample_id_column,
    )


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
    Sanitise score matrices by replacing NaN/Inf values with large finite sentinels.

    This prevents downstream routines (e.g. ``roc_curve``) from failing while still
    ensuring missing scores are ignored after activation/thresholding.
    """
    array = np.asarray(scores, dtype=np.float64)
    np.nan_to_num(
        array,
        nan=MISSING_SCORE_FILL_VALUE,
        posinf=SCORE_CLIP_ABS,
        neginf=MISSING_SCORE_FILL_VALUE,
        copy=False,
    )
    np.clip(array, -SCORE_CLIP_ABS, SCORE_CLIP_ABS, out=array)
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


def _derive_class_name(column: str) -> str:
    lower = column.lower()
    for prefix in WIDE_PREFIXES:
        if lower.startswith(prefix):
            name = column[len(prefix) :].strip()
            if name:
                return name
    return column


def _build_wide_map_from_columns(columns: Sequence[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for column in columns:
        class_name = _derive_class_name(column)
        if class_name in mapping:
            raise ValueError(
                f"Ambiguous class mapping detected for column '{column}'. "
                f"The derived class name '{class_name}' is duplicated."
            )
        mapping[class_name] = column
    return mapping


def _auto_detect_wide_columns(df: pd.DataFrame, true_label_column: str) -> tuple[str, ...]:
    detected: list[str] = []
    for column in df.columns:
        if column == true_label_column:
            continue
        lower = column.lower()
        for prefix in WIDE_PREFIXES:
            if lower.startswith(prefix):
                detected.append(column)
                break
    return tuple(detected)


def _validate_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in dataframe.")
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


__all__ = [
    "ColumnCandidates",
    "ColumnSelection",
    "DataMetadata",
    "ROW_INDEX_COLUMN",
    "handle_missing_scores",
    "infer_column_candidates",
    "load_csv",
    "prepare_score_matrix",
    "validate_data",
    "get_majority_class",
]