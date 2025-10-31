import numpy as np
import pytest

from app.utils import activations


def test_softmax_rows_sum_to_one():
    logits = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
    result = activations.softmax(logits)
    row_sums = result.sum(axis=1)
    assert np.allclose(row_sums, 1.0)


def test_softmax_numerical_stability_with_large_values():
    logits = np.array([[1000.0, 1001.0, 1002.0]])
    result = activations.softmax(logits)
    expected = np.exp([0.0, 1.0, 2.0]) / np.sum(np.exp([0.0, 1.0, 2.0]))
    assert np.allclose(result[0], expected, atol=1e-6)


def test_sigmoid_extreme_values():
    x = np.array([-100.0, 0.0, 100.0])
    result = activations.sigmoid(x)
    assert np.isclose(result[0], 0.0, atol=1e-12)
    assert np.isclose(result[1], 0.5, atol=1e-12)
    assert np.isclose(result[2], 1.0, atol=1e-12)


def test_sigmoid_5_scaling():
    x = np.array([-5.0, 0.0, 5.0])
    base = activations.sigmoid(x / 5.0)
    result = activations.sigmoid_5(x)
    assert np.allclose(result, base)


def test_apply_activation_passthrough_none():
    scores = np.array([[0.2, 0.8], [0.6, 0.4]])
    result = activations.apply_activation(scores, "none")
    assert np.allclose(result, scores)


def test_apply_activation_softmax_and_sigmoid_variants():
    scores = np.array([[0.3, -0.7, 1.2]])
    softmax_result = activations.apply_activation(scores, "softmax")
    assert np.allclose(softmax_result.sum(axis=1), 1.0)

    sigmoid_result = activations.apply_activation(scores, "sigmoid")
    expected = activations.sigmoid(scores)
    assert np.allclose(sigmoid_result, expected)

    sigmoid5_result = activations.apply_activation(scores, "sigmoid_5")
    expected5 = activations.sigmoid_5(scores)
    assert np.allclose(sigmoid5_result, expected5)


def test_apply_activation_invalid_raises_value_error():
    scores = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError):
        activations.apply_activation(scores, "unknown")