from pd_model.monitoring.psi import calculate_psi
import numpy as np
import pytest


def test_psi_identical_distributions():
    expected = np.array([10, 20, 30, 40, 50] * 20)
    actual = expected.copy()

    psi = calculate_psi(expected, actual)

    assert psi == pytest.approx(0.0, abs=1e-8)


def test_psi_different_distributions():
    expected = np.linspace(0, 1, 1000)
    actual = np.linspace(5, 6, 1000)

    psi = calculate_psi(expected, actual)

    assert psi > 0


def test_psi_empty_data():
    expected = np.array([])
    actual = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        calculate_psi(expected, actual)


def test_psi_nan_values():
    expected = np.array([1, 2, np.nan, 4])
    actual = np.array([1, 2, 3, 4])

    with pytest.raises(ValueError):
        calculate_psi(expected, actual)