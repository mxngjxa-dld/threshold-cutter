"""
Plotting helpers for visualising evaluation outputs inside Streamlit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import PowerNorm
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix

sns.set_theme(context="notebook", style="whitegrid")


DEFAULT_FONT_FAMILY = "Menlo, monospace"
FIGURE_DIR = Path("app") / "outputs" / "figures"


def _ensure_output_dir(path: Path | None) -> Path | None:
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _snake_case(name: str) -> str:
    safe = "".join(char if char.isalnum() else "_" for char in name.lower())
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_")


def plot_confusion_matrix_raw(
    y_true: Sequence,
    y_pred: Sequence,
    classes: Sequence[str],
    f1_macro: float | None = None,
    output_path: str | Path | None = None,
    dpi: int = 300,
):
    """
    Render a confusion matrix heatmap styled for the Streamlit app.
    """
    classes = list(classes)
    matrix = confusion_matrix(y_true, y_pred, labels=classes)

    plt.rcParams["font.family"] = DEFAULT_FONT_FAMILY

    fig, ax = plt.subplots(figsize=(12, 10))
    norm = PowerNorm(gamma=0.2)
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        norm=norm,
        cbar=True,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
    )

    ax.set_xlabel("Predicted class", fontweight="bold")
    ax.set_ylabel("True class", fontweight="bold")
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes, rotation=0)

    title = "Confusion Matrix"
    if f1_macro is not None:
        title += f" (F1-Macro: {f1_macro:.4f})"
    ax.set_title(title, fontweight="bold")

    fig.tight_layout()

    if output_path:
        path = _ensure_output_dir(Path(output_path))
        fig.savefig(path, dpi=dpi, bbox_inches="tight")

    return fig


def format_roc_figure(ax: plt.Axes, title: str | None = None) -> None:
    """
    Apply consistent styling to ROC axis.
    """
    plt.rcParams["font.family"] = DEFAULT_FONT_FAMILY
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    if title:
        ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right")


def plot_per_class_roc(
    roc_data: Mapping[str, tuple[np.ndarray, np.ndarray]],
    classes: Iterable[str],
    filter_classes: Iterable[str] | None = None,
    layout: tuple[int, int] | None = None,
    output_prefix: str | None = None,
    dpi: int = 300,
) -> dict[str, plt.Figure]:
    """
    Create individual or multi-panel ROC curves from pre-computed ROC data.

    Parameters
    ----------
    roc_data:
        Mapping of class label -> (fpr array, tpr array).
    classes:
        Iterable of class labels, used for ordering.
    filter_classes:
        Optional subset of classes to include.
    layout:
        Optional (rows, cols) for multi-panel display. If None, auto-calculated.
    output_prefix:
        If provided, saves figures into outputs/figures with the prefix.
    """
    class_order = [cls for cls in classes if cls in roc_data]
    if filter_classes is not None:
        selected = set(filter_classes)
        class_order = [cls for cls in class_order if cls in selected]

    if not class_order:
        raise ValueError("No classes available for ROC plotting with given filter.")

    figures: dict[str, plt.Figure] = {}
    needs_save = output_prefix is not None
    output_prefix = output_prefix or "roc"

    if layout is None:
        cols = min(3, len(class_order))
        rows = int(np.ceil(len(class_order) / cols))
    else:
        rows, cols = layout

    plt.rcParams["font.family"] = DEFAULT_FONT_FAMILY
    multi_fig = None
    multi_axes = None
    create_multi = len(class_order) > 1

    if create_multi:
        multi_fig, multi_axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        axes_iter = np.ravel(multi_axes)
    else:
        axes_iter = [plt.subplots(figsize=(6, 5))[1]]

    for idx, cls in enumerate(class_order):
        fpr, tpr = roc_data[cls]
        if create_multi:
            ax = axes_iter[idx]
        else:
            ax = axes_iter[0]

        ax.plot(fpr, tpr, label=f"{cls} ROC")
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", alpha=0.7)
        format_roc_figure(ax, f"ROC Curve — {cls}")

        fig = ax.get_figure()
        figures[cls] = fig

        if needs_save:
            filename = (
                FIGURE_DIR / f"{_snake_case(output_prefix)}_{_snake_case(cls)}.png"
            )
            path = _ensure_output_dir(filename)
            fig.savefig(path, dpi=dpi, bbox_inches="tight")

    if create_multi:
        for idx in range(len(class_order), len(axes_iter)):
            axes_iter[idx].axis("off")
        multi_fig.tight_layout()
        if needs_save:
            filename = FIGURE_DIR / f"{_snake_case(output_prefix)}_combined.png"
            path = _ensure_output_dir(filename)
            multi_fig.savefig(path, dpi=dpi, bbox_inches="tight")
        figures["combined"] = multi_fig

    return figures


def create_inline_roc_display(
    roc_df,
    classes: Iterable[str],
    filter_classes: Iterable[str] | None = None,
    output_prefix: str | None = None,
) -> dict[str, plt.Figure]:
    """
    Prepare ROC figures for Streamlit display.

    Parameters
    ----------
    roc_df:
        DataFrame with index of class names and a column ``roc_curve`` containing
        (fpr, tpr) arrays. Additional columns (e.g., AUC) are ignored.
    classes:
        Ordered iterable of class names.
    filter_classes:
        Optional subset of classes to visualise.
    output_prefix:
        Optional prefix for saved figures.
    """
    if "roc_curve" not in roc_df.columns:
        raise ValueError("roc_df must contain a 'roc_curve' column.")

    roc_data = {
        cls: roc_df.loc[cls, "roc_curve"]
        for cls in roc_df.index
        if isinstance(roc_df.loc[cls, "roc_curve"], tuple)
    }

    figures = plot_per_class_roc(
        roc_data=roc_data,
        classes=classes,
        filter_classes=filter_classes,
        output_prefix=output_prefix,
    )
    return figures


__all__ = [
    "create_inline_roc_display",
    "format_roc_figure",
    "plot_confusion_matrix_raw",
    "plot_per_class_roc",
]
