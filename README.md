# Multiclass Threshold Tuner

Interactive Streamlit application for tuning multiclass classification thresholds, visualising performance metrics, and exporting analysis artifacts.

This README summarises the current implementation, how to run the app, how to exercise the testing stack, and how to stress-test the system with the synthetic data tools bundled in this repository.

## Key Capabilities

- Upload logits or probability CSVs in "wide" or "long" form and automatically validate schema.
- Apply activation functions (none, softmax, sigmoid, sigmoid_5) with per-class thresholding and fallbacks.
- Compute confusion matrices, ROC/PR curves, aggregated metrics, and Youden's J statistics with robust handling for zero-support classes.
- Export metrics, ROC figures, and prediction CSVs with consistent timestamps.
- Generate synthetic datasets with configurable edge cases (class imbalance, near-threshold noise, missing scores, extremes, duplicates).

## Repository Layout

- [`app/main.py`](app/main.py) — Streamlit entrypoint and UI logic.
- [`app/utils/metrics.py`](app/utils/metrics.py) — Metric aggregation, ROC/Youden handling, and summary dataclasses.
- [`app/utils/thresholds.py`](app/utils/thresholds.py) — Threshold application and optimal Youden threshold computation.
- [`app/utils/data_io.py`](app/utils/data_io.py) — CSV loading, schema validation, and score matrix preparation.
- [`app/utils/plots.py`](app/utils/plots.py) — Confusion matrix and ROC figure generation.
- [`app/utils/exports.py`](app/utils/exports.py) — Metrics, figures, and predictions export helpers.
- [`tests/test_metrics.py`](tests/test_metrics.py) & [`tests/test_thresholds.py`](tests/test_thresholds.py) — Focused regression coverage for metrics and threshold edge cases.
- [`tests/generate_synthetic_data.py`](tests/generate_synthetic_data.py) — CLI generator for synthetic datasets with optional edge-case injection.

## Environment Setup

This project uses [uv](https://github.com/astral-sh/uv) to manage Python execution. Install uv if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install the project dependencies (uv will create an isolated environment automatically):

```bash
uv pip install -r requirements.txt
```

## Running the Streamlit Application

1. Ensure the dependency environment is prepared (see above).
2. Start the application with uv:

```bash
uv run streamlit run app/main.py
```

3. Open the URL shown in the console (defaults to http://localhost:8502/) to interact with the UI.

### Streamlit Workflow Checklist

Use this checklist while validating the UI with large synthetic datasets (e.g. `tests/synthetic_data/synthetic_50k_wide.csv` and `tests/synthetic_data/synthetic_50k_long.csv`):

- Data loading succeeds for both wide and long formats (auto delimiter detection works).
- Activation selection (none, softmax, sigmoid, sigmoid_5) updates the probability interpretation.
- Global and per-class threshold sliders respond immediately, including automatic Youden-derived modes.
- Metrics cards update accuracy / macro metrics; micro & macro AUC render as expected for missing-support classes.
- Confusion matrix, ROC overlays, and PR curves render without errors at 50k+ rows.
- Exports (metrics, ROC figures, confusion matrix, predictions) write files into [`app/outputs/`](app/outputs/) without exceptions.

## Data Format

### Wide Format

- Required columns: `true_label`, `predicted_class`, and one or more `logit_{class_name}` columns.
- Optional columns: sample identifiers or metadata (retained in exports).
- Example header: `sample_id,true_label,logit_class_00,logit_class_01,...`.

### Long Format

- Required columns: `true_label`, `predicted_category`, `logit_score`.
- Optional columns: `sample_id` (used to reconstruct wide matrices) plus any additional metadata.
- Each row encodes a (sample, class) pair; the loader pivots the data into wide matrices automatically.

Data validation enforces non-empty datasets, consistent class coverage, and infers class order from observed columns. Missing logit values are imputed to `-inf` before activation to avoid threshold leakage.

## Metrics & Threshold Handling Notes

- [`app/utils/metrics.py`](app/utils/metrics.py) gracefully returns NaN AUC/Youden values for classes without positive or negative support.
- [`app/utils/metrics.py`](app/utils/metrics.py) aggregates per-class support via `support_pos + support_neg` and casts AUC columns to floats for consistent downstream rendering.
- [`app/utils/thresholds.py`](app/utils/thresholds.py) provides Youden-optimal thresholds while returning NaN / sentinel statistics for missing-support classes, ensuring downstream tables remain stable.

These behaviours are covered by unit tests so regressions surface quickly.

## Synthetic Data Generation

The generator creates both wide and long CSVs with configurable edge-case controls.

### Quick Example (10k samples)

```bash
uv run python tests/generate_synthetic_data.py --num-samples 10000 --output-dir tests/synthetic_data --prefix synthetic_10k
```

### Required Arguments

- `--num-samples` *(int, default 5000)* — Number of base samples to generate (duplicates may add more).
- `--num-classes` *(int, default 10)* — Distinct class labels (named `class_00`, `class_01`, ...).

### Edge-Case Controls

- `--imbalance-factor` — Geometric decay controlling class prevalence (0 < factor ≤ 1).
- `--near-threshold-proportion` — Proportion forced near decision boundaries.
- `--label-noise` — Fraction of labels randomly flipped to alternate classes.
- `--missing-score-ratio` — Fraction of logit cells set to NaN.
- `--extreme-score-ratio` / `--extreme-score-scale` — Inject extreme logits to stress activation clipping.
- `--duplicate-ratio` — Duplicates rows to replicate repeated samples.

Generated files are written to the supplied `--output-dir` as `{prefix}_wide.csv` and `{prefix}_long.csv`.

## Testing

Run the focused regression suite with uv:

```bash
uv run pytest tests/test_metrics.py tests/test_thresholds.py -v
```

Common targeted invocations:

```bash
uv run pytest tests/test_metrics.py::test_per_class_roc_and_j_handles_missing_support -v
uv run pytest tests/test_metrics.py::test_create_metrics_summary -v
uv run pytest tests/test_thresholds.py::test_compute_optimal_thresholds_handles_missing_support -v
uv run pytest tests/test_thresholds.py::test_select_predicted_class_with_no_valid_scores -v
```

## Performance Tips

- Use activation-aware threshold bounds (the UI clips sliders between [0,1] for probability activations and [-10,10] for raw logits).
- The Streamlit pipeline caches CSV parsing, dataset preparation, and prediction/metric computation to keep update times sub-second even for 50k+ rows.
- When experimenting with massive datasets, close unused browser tabs and monitor console memory usage; the generator can exceed requested rows because of duplicates.

## Troubleshooting

| Symptom | Likely Cause | Remedy |
| --- | --- | --- |
| CSV upload fails with delimiter error | Nonstandard delimiter | Supply `Delimiter override` in the sidebar |
| Metrics display NaN for AUC/Youden | Class lacks positive/negative support | This is expected; thresholds fall back to defaults |
| Exports missing | Output directory unwritable | Ensure [`app/outputs/`](app/outputs/) is writable or adjust permissions |
| Streamlit UI sluggish | Large figures queued | Close redundant plots or restart the app to clear cache |

## Roadmap

Future enhancements noted in [`PROJECT_SPEC.md`](PROJECT_SPEC.md) include precision-recall curve expansions, Plotly visualisations, and extended documentation screenshots.

## License

MIT License. See [`LICENSE`](LICENSE) if present, or customise as required for your deployment.

---

For questions or contributions, open an issue or submit a pull request.