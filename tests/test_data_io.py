import numpy as np
import pandas as pd

from app.utils.data_io import (
    ColumnCandidates,
    ColumnSelection,
    infer_column_candidates,
    prepare_score_matrix,
    validate_data,
)


def _assert_column_candidates(candidate: ColumnCandidates) -> None:
    assert isinstance(candidate.true_label_default, str)
    assert candidate.true_label_default in candidate.true_label_options
    assert candidate.default_format in ("wide", "long")


def test_infer_column_candidates_wide_defaults() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "ground_truth": ["cat", "dog"],
            "score_cat": [0.9, 0.1],
            "score_dog": [0.1, 0.8],
        }
    )

    candidates = infer_column_candidates(df)

    _assert_column_candidates(candidates)
    assert candidates.default_format == "wide"
    assert candidates.true_label_default == "ground_truth"
    assert set(candidates.wide_score_default) == {"score_cat", "score_dog"}
    assert "score_cat" in candidates.wide_score_options
    assert "score_dog" in candidates.wide_score_options


def test_validate_data_with_explicit_wide_selection() -> None:
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "actual": ["positive", "negative", "positive"],
            "prob_positive": [0.7, 0.2, 0.9],
            "prob_negative": [0.3, 0.8, 0.1],
        }
    )

    selection = ColumnSelection(
        format="wide",
        true_label="actual",
        wide_score_columns=("prob_positive", "prob_negative"),
        sample_id_column="customer_id",
    )
    metadata = validate_data(df, column_selection=selection)

    assert metadata.format == "wide"
    assert metadata.true_label_column == "actual"
    assert metadata.sample_id_column == "customer_id"
    assert metadata.class_to_column == {
        "positive": "prob_positive",
        "negative": "prob_negative",
    }

    scores, index = prepare_score_matrix(df, metadata=metadata, return_index=True)
    assert scores.shape == (3, 2)
    np.testing.assert_array_equal(index.to_numpy(), df.index.to_numpy())


def test_validate_data_with_explicit_long_selection() -> None:
    df = pd.DataFrame(
        {
            "record_id": [101, 101, 202],
            "target": ["spam", "spam", "ham"],
            "predicted_label": ["spam", "ham", "spam"],
            "probability": [0.9, 0.1, 0.6],
        }
    )

    selection = ColumnSelection(
        format="long",
        true_label="target",
        long_class_column="predicted_label",
        long_score_column="probability",
        sample_id_column="record_id",
    )
    metadata = validate_data(df, column_selection=selection)

    assert metadata.format == "long"
    assert metadata.true_label_column == "target"
    assert metadata.long_class_column == "predicted_label"
    assert metadata.long_score_column == "probability"
    assert metadata.sample_id_column == "record_id"
    assert set(metadata.classes) == {"spam", "ham"}

    scores, index = prepare_score_matrix(df, metadata=metadata, return_index=True)
    assert scores.shape == (2, len(metadata.classes))
    assert set(index.to_numpy()) == {101, 202}