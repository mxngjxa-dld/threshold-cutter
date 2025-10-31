"""
Synthetic data generator for the multiclass threshold tuner project.

Produces both "wide" (single row per sample with logit columns) and "long"
(one row per sample-class pair) CSV files with configurable dataset size,
class count, and optional edge-case injections targeted at exercising the UI.
Edge cases include heavy class imbalance, scores near decision thresholds,
missing values, label noise, extreme logits, and duplicated samples.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

LOGIT_PREFIX = "logit_"


@dataclass(frozen=True)
class SyntheticDataset:
    wide: pd.DataFrame
    long: pd.DataFrame
    classes: tuple[str, ...]
    parameters: dict[str, int | float | str | None] | None = None


def _build_class_labels(num_classes: int) -> tuple[str, ...]:
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2 to form a multiclass problem.")
    return tuple(f"class_{idx:02d}" for idx in range(num_classes))


def _build_class_weights(
    num_classes: int,
    imbalance_factor: float = 0.65,
) -> np.ndarray:
    imbalance_factor = float(imbalance_factor)
    if not (0.0 < imbalance_factor <= 1.0):
        raise ValueError("imbalance_factor must lie within (0, 1].")
    if np.isclose(imbalance_factor, 1.0):
        return np.full(num_classes, 1.0 / num_classes, dtype=np.float64)
    weights = imbalance_factor ** np.arange(num_classes, dtype=np.float64)
    weights /= weights.sum()
    return weights


def _validate_ratio(name: str, value: float, *, max_value: float = 1.0) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite value.")
    if value < 0.0 or value > max_value:
        raise ValueError(f"{name} must be within [0, {max_value}].")
    return value


def _generate_logits_internal(
    num_samples: int,
    num_classes: int,
    rng: np.random.Generator,
    imbalance_factor: float,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if num_samples <= 0:
        raise ValueError("num_samples must be a positive integer.")
    classes = _build_class_labels(num_classes)
    weights = _build_class_weights(num_classes, imbalance_factor)
    logits = rng.normal(loc=0.0, scale=3.0, size=(num_samples, len(classes)))
    y_true = rng.choice(classes, size=num_samples, p=weights)
    return logits.astype(np.float64), np.asarray(y_true, dtype=object), classes


def generate_synthetic_logits(
    num_samples: int,
    num_classes: int,
    seed: int | None = 42,
    imbalance_factor: float = 0.65,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    rng = np.random.default_rng(seed)
    return _generate_logits_internal(num_samples, num_classes, rng, imbalance_factor)


def _inject_near_threshold(
    logits: np.ndarray,
    rng: np.random.Generator,
    proportion: float,
    jitter_scale: float = 0.25,
) -> None:
    if proportion <= 0.0 or logits.size == 0:
        return
    num_rows = logits.shape[0]
    count = max(1, int(round(num_rows * proportion)))
    count = min(count, num_rows)
    indices = rng.choice(num_rows, size=count, replace=False)
    jitter = rng.normal(loc=0.0, scale=jitter_scale, size=(count, logits.shape[1]))
    logits[indices] = jitter


def _inject_extreme_scores(
    logits: np.ndarray,
    rng: np.random.Generator,
    proportion: float,
    scale: float,
) -> None:
    if proportion <= 0.0 or logits.size == 0:
        return
    if scale <= 0.0:
        raise ValueError("extreme_score_scale must be positive.")
    total = logits.size
    count = max(1, int(round(total * proportion)))
    count = min(count, total)
    flat_indices = rng.choice(total, size=count, replace=False)
    rows, cols = np.unravel_index(flat_indices, logits.shape)
    signs = rng.choice([-1.0, 1.0], size=count)
    magnitudes = rng.uniform(low=scale * 0.75, high=scale, size=count)
    logits[rows, cols] = signs * magnitudes


def _apply_label_noise(
    y_true: np.ndarray,
    classes: Iterable[str],
    noise_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if noise_rate <= 0.0 or y_true.size == 0:
        return y_true
    classes_array = np.asarray(tuple(classes), dtype=object)
    if classes_array.size <= 1:
        return y_true
    num_samples = y_true.size
    count = max(1, int(round(num_samples * noise_rate)))
    count = min(count, num_samples)
    indices = rng.choice(num_samples, size=count, replace=False)
    mutated = y_true.copy()
    for idx in indices:
        current = mutated[idx]
        candidates = classes_array[classes_array != current]
        if candidates.size == 0:
            continue
        mutated[idx] = rng.choice(candidates)
    return mutated


def _build_wide_dataframe(
    logits: np.ndarray,
    y_true: np.ndarray,
    classes: Iterable[str],
) -> pd.DataFrame:
    class_list = list(classes)
    score_columns = [f"{LOGIT_PREFIX}{cls}" for cls in class_list]
    scores_df = pd.DataFrame(logits, columns=score_columns)
    wide_df = pd.DataFrame({"true_label": pd.Series(y_true, dtype="string")})
    wide_df = pd.concat([wide_df, scores_df], axis=1)
    predicted_idx = np.argmax(logits, axis=1)
    classes_array = np.asarray(class_list, dtype=object)
    predicted = classes_array[predicted_idx]
    wide_df["predicted_class"] = pd.Series(predicted, dtype="string")
    return wide_df


def _inject_missing_scores(
    wide_df: pd.DataFrame,
    classes: Iterable[str],
    ratio: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if ratio <= 0.0 or wide_df.empty:
        return wide_df
    value_columns = [f"{LOGIT_PREFIX}{cls}" for cls in classes]
    total_cells = len(value_columns) * len(wide_df)
    if total_cells == 0:
        return wide_df
    count = max(1, int(round(total_cells * ratio)))
    count = min(count, total_cells)
    wide = wide_df.copy()
    index_array = wide.index.to_numpy()
    col_choices = np.arange(len(value_columns))
    rows = rng.choice(index_array, size=count, replace=True)
    cols = rng.choice(col_choices, size=count, replace=True)
    for row, col_idx in zip(rows, cols):
        wide.at[row, value_columns[col_idx]] = np.nan
    return wide


def _append_duplicates(
    wide_df: pd.DataFrame,
    ratio: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if ratio <= 0.0 or wide_df.empty:
        return wide_df
    count = max(1, int(round(len(wide_df) * ratio)))
    random_state = int(rng.integers(0, 2**32 - 1))
    duplicates = wide_df.sample(
        n=count,
        replace=len(wide_df) < count,
        random_state=random_state,
    ).copy()
    return pd.concat([wide_df, duplicates], ignore_index=True)


def _shuffle_dataframe(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    if len(df) <= 1:
        return df.reset_index(drop=True)
    permutation = rng.permutation(len(df))
    return df.iloc[permutation].reset_index(drop=True)


def _attach_sample_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    df.insert(0, "sample_id", np.arange(len(df), dtype=np.int64))
    base_columns = ["sample_id"]
    for column in ("true_label", "predicted_class"):
        if column in df.columns:
            base_columns.append(column)
    score_columns = [col for col in df.columns if col.startswith(LOGIT_PREFIX)]
    remaining = [col for col in df.columns if col not in base_columns + score_columns]
    ordered_columns = base_columns + score_columns + remaining
    return df.loc[:, ordered_columns]


def _build_long_dataframe(
    wide_df: pd.DataFrame,
    classes: Iterable[str],
) -> pd.DataFrame:
    value_vars = [f"{LOGIT_PREFIX}{cls}" for cls in classes]
    long_df = wide_df.melt(
        id_vars=["sample_id", "true_label", "predicted_class"],
        value_vars=value_vars,
        var_name="predicted_category",
        value_name="logit_score",
    )
    long_df["predicted_category"] = (
        long_df["predicted_category"].str[len(LOGIT_PREFIX) :].astype("string")
    )
    long_df["logit_score"] = long_df["logit_score"].astype(np.float64)
    return long_df


def generate_synthetic_dataset(
    num_samples: int = 5000,
    num_classes: int = 10,
    seed: int | None = 42,
    imbalance_factor: float = 0.65,
    near_threshold_proportion: float = 0.08,
    label_noise: float = 0.02,
    missing_score_ratio: float = 0.01,
    duplicate_ratio: float = 0.05,
    extreme_score_ratio: float = 0.01,
    extreme_score_scale: float = 12.0,
) -> SyntheticDataset:
    _validate_ratio("near_threshold_proportion", near_threshold_proportion)
    _validate_ratio("label_noise", label_noise, max_value=0.5)
    _validate_ratio("missing_score_ratio", missing_score_ratio, max_value=0.5)
    _validate_ratio("duplicate_ratio", duplicate_ratio, max_value=1.0)
    _validate_ratio("extreme_score_ratio", extreme_score_ratio)
    if extreme_score_scale <= 0.0:
        raise ValueError("extreme_score_scale must be positive.")
    rng = np.random.default_rng(seed)
    logits, y_true, classes = _generate_logits_internal(
        num_samples=num_samples,
        num_classes=num_classes,
        rng=rng,
        imbalance_factor=imbalance_factor,
    )
    _inject_near_threshold(logits, rng, near_threshold_proportion)
    _inject_extreme_scores(logits, rng, extreme_score_ratio, extreme_score_scale)
    y_true = _apply_label_noise(y_true, classes, label_noise, rng)
    wide_df = _build_wide_dataframe(logits, y_true, classes)
    wide_df = _inject_missing_scores(wide_df, classes, missing_score_ratio, rng)
    wide_df = _append_duplicates(wide_df, duplicate_ratio, rng)
    wide_df = _shuffle_dataframe(wide_df, rng)
    wide_df = _attach_sample_ids(wide_df)
    wide_df["true_label"] = wide_df["true_label"].astype("string")
    wide_df["predicted_class"] = wide_df["predicted_class"].astype("string")
    long_df = _build_long_dataframe(wide_df, classes)
    metadata = {
        "requested_num_samples": int(num_samples),
        "actual_num_samples": int(len(wide_df)),
        "num_classes": int(len(classes)),
        "seed": seed,
        "imbalance_factor": float(imbalance_factor),
        "near_threshold_proportion": float(near_threshold_proportion),
        "label_noise": float(label_noise),
        "missing_score_ratio": float(missing_score_ratio),
        "duplicate_ratio": float(duplicate_ratio),
        "extreme_score_ratio": float(extreme_score_ratio),
        "extreme_score_scale": float(extreme_score_scale),
    }
    return SyntheticDataset(
        wide=wide_df,
        long=long_df,
        classes=tuple(classes),
        parameters=metadata,
    )


def save_dataset(
    dataset: SyntheticDataset,
    output_dir: Path,
    prefix: str = "synthetic",
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    wide_path = output_dir / f"{prefix}_wide.csv"
    long_path = output_dir / f"{prefix}_long.csv"
    dataset.wide.to_csv(wide_path, index=False)
    dataset.long.to_csv(long_path, index=False)
    return wide_path, long_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic multiclass classification datasets."
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=5000,
        help="Number of samples to generate (default: 5000).",
    )
    parser.add_argument(
        "-k",
        "--num-classes",
        type=int,
        default=10,
        help="Number of classes (default: 10).",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--imbalance-factor",
        type=float,
        default=0.65,
        help="Geometric decay factor controlling class imbalance (0,1] (default: 0.65).",
    )
    parser.add_argument(
        "--near-threshold-proportion",
        type=float,
        default=0.08,
        help="Proportion of rows nudged close to the decision threshold (default: 0.08).",
    )
    parser.add_argument(
        "--label-noise",
        type=float,
        default=0.02,
        help="Fraction of labels to randomly flip to alternate classes (default: 0.02).",
    )
    parser.add_argument(
        "--missing-score-ratio",
        type=float,
        default=0.01,
        help="Fraction of score cells set to NaN to emulate missing data (default: 0.01).",
    )
    parser.add_argument(
        "--duplicate-ratio",
        type=float,
        default=0.05,
        help="Fraction of samples to duplicate for robustness checks (default: 0.05).",
    )
    parser.add_argument(
        "--extreme-score-ratio",
        type=float,
        default=0.01,
        help="Fraction of logits replaced with extreme values (default: 0.01).",
    )
    parser.add_argument(
        "--extreme-score-scale",
        type=float,
        default=12.0,
        help="Magnitude of extreme logits injected into the dataset (default: 12.0).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("tests") / "synthetic_data",
        help="Directory to save the generated CSV files.",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default="synthetic_dataset",
        help="Filename prefix for saved outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = generate_synthetic_dataset(
        num_samples=args.num_samples,
        num_classes=args.num_classes,
        seed=args.seed,
        imbalance_factor=args.imbalance_factor,
        near_threshold_proportion=args.near_threshold_proportion,
        label_noise=args.label_noise,
        missing_score_ratio=args.missing_score_ratio,
        duplicate_ratio=args.duplicate_ratio,
        extreme_score_ratio=args.extreme_score_ratio,
        extreme_score_scale=args.extreme_score_scale,
    )
    wide_path, long_path = save_dataset(dataset, args.output_dir, prefix=args.prefix)
    meta = dataset.parameters or {}
    generated = meta.get("actual_num_samples", len(dataset.wide))
    classes = len(dataset.classes)
    print(f"Generated {generated} samples across {classes} classes.")
    print(f"Saved wide dataset to {wide_path}")
    print(f"Saved long dataset to {long_path}")


__all__ = [
    "SyntheticDataset",
    "generate_synthetic_logits",
    "generate_synthetic_dataset",
    "save_dataset",
    "parse_args",
    "main",
]


if __name__ == "__main__":
    main()