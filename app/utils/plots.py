"""
Plotting helpers for visualising evaluation outputs inside Streamlit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure as MplFigure
import numpy as np
import seaborn as sns
from matplotlib.colors import PowerNorm
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import auc, confusion_matrix

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
        if path is not None:  # Type guard
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
) -> dict[str, MplFigure]:
    """
    Create individual or multi-panel ROC curves from pre-computed ROC data.

    Uses the tab20 colour palette (up to 20 classes) with a rainbow fallback
    and appends AUC scores to legend entries for improved readability.
    """
    class_order = [cls for cls in classes if cls in roc_data]
    if filter_classes is not None:
        selected = set(filter_classes)
        class_order = [cls for cls in class_order if cls in selected]

    if not class_order:
        raise ValueError("No classes available for ROC plotting with given filter.")

    num_classes = len(class_order)
    if num_classes <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:num_classes]
    else:
        colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))
    color_map = {cls: colors[idx] for idx, cls in enumerate(class_order)}

    figures: dict[str, MplFigure] = {}
    needs_save = output_prefix is not None
    output_prefix = output_prefix or "roc"

    if layout is None:
        cols = min(3, len(class_order))
        rows = int(np.ceil(len(class_order) / cols))
    else:
        rows, cols = layout

    plt.rcParams["font.family"] = DEFAULT_FONT_FAMILY
    multi_fig: MplFigure | None = None
    multi_axes = None
    create_multi = len(class_order) > 1

    if create_multi:
        multi_fig, multi_axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        axes_iter = np.ravel(multi_axes)
    else:
        _, ax = plt.subplots(figsize=(6, 5))
        axes_iter = [ax]

    for idx, cls in enumerate(class_order):
        fpr, tpr = roc_data[cls]
        ax = axes_iter[idx] if create_multi else axes_iter[0]

        auc_score = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            color=color_map[cls],
            linewidth=2,
            label=f"{cls} (AUC={auc_score:.3f})",
        )
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", alpha=0.7)
        format_roc_figure(ax, f"ROC Curve — {cls}")

        if not create_multi:
            fig = ax.get_figure()
            if fig is not None and isinstance(fig, MplFigure):
                figures[str(cls)] = fig
                if needs_save:
                    filename = (
                        FIGURE_DIR
                        / f"{_snake_case(output_prefix)}_{_snake_case(cls)}.png"
                    )
                    path = _ensure_output_dir(filename)
                    if path is not None:
                        fig.savefig(str(path), dpi=dpi, bbox_inches="tight")

    if create_multi:
        for idx in range(len(class_order), len(axes_iter)):
            axes_iter[idx].axis("off")
        if multi_fig is not None and isinstance(multi_fig, MplFigure):
            multi_fig.tight_layout()
            if needs_save:
                filename = FIGURE_DIR / f"{_snake_case(output_prefix)}_combined.png"
                path = _ensure_output_dir(filename)
                if path is not None:
                    multi_fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
            figures["combined"] = multi_fig

    return figures


def plotcombinedrocallmodels(
    models_roc_data: Mapping[str, Mapping[str, tuple[np.ndarray, np.ndarray]]],
    model_names: Sequence[str],
    classes: Sequence[str],
    filterclasses: Sequence[str] | None = None,
    outputpath: str | Path | None = None,
    dpi: int = 300,
) -> plt.Figure | None:
    """
    Render side-by-side ROC curves for multiple models, reusing class colours.

    Parameters
    ----------
    models_roc_data:
        Mapping of model name -> mapping of class -> (fpr, tpr) arrays.
    model_names:
        Ordered collection of model identifiers to display.
    classes:
        All available class labels.
    filterclasses:
        Optional subset of ``classes`` to visualise.
    outputpath:
        Optional path where the combined figure should be written.
    dpi:
        Resolution for the saved figure.

    Returns
    -------
    plt.Figure | None
        The combined Matplotlib Figure or ``None`` if no curves could be plotted.
    """
    if filterclasses is not None:
        selected_classes = [cls for cls in classes if cls in filterclasses]
    else:
        selected_classes = list(classes)

    if not selected_classes:
        raise ValueError("No classes available for ROC plotting with given filter.")
    if not model_names:
        raise ValueError("At least one model name is required for ROC plotting.")

    for model in model_names:
        if model not in models_roc_data:
            raise ValueError(f"Model '{model}' not found in ROC data.")

    num_classes = len(selected_classes)
    if num_classes <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:num_classes]
    else:
        colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))
    color_map = {cls: colors[idx] for idx, cls in enumerate(selected_classes)}

    plt.rcParams["font.family"] = DEFAULT_FONT_FAMILY
    num_models = len(model_names)
    fig, axes = plt.subplots(1, num_models, figsize=(6.5 * num_models, 6))

    if num_models == 1:
        axes = [axes]

    fig.suptitle("ROC Curves per Category for Each Model", fontsize=16, y=1.02)
    has_curve = False

    for model_idx, model in enumerate(model_names):
        ax = axes[model_idx]
        roc_data = models_roc_data[model]

        for cls in selected_classes:
            if cls not in roc_data:
                continue
            fpr, tpr = roc_data[cls]
            auc_score = auc(fpr, tpr)
            ax.plot(
                fpr,
                tpr,
                color=color_map[cls],
                linewidth=2,
                label=f"{cls} (AUC={auc_score:.3f})",
            )
            has_curve = True

        ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random (AUC=0.5)")
        format_roc_figure(ax, f"Model {model}")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if not has_curve:
        plt.close(fig)
        return None

    if outputpath is not None:
        path = _ensure_output_dir(Path(outputpath))
        if path is not None:
            fig.savefig(path, dpi=dpi, bbox_inches="tight")

    return fig


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
    "plotcombinedrocallmodels",
]
