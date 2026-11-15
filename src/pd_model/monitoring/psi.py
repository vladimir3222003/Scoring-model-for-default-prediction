import numpy as np


def calculate_psi(expected, actual, bins=10, epsilon=1e-6):

    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    if expected.ndim != 1 or actual.ndim != 1:
        raise ValueError("Входные данные должны быть одномерными")

    if expected.size == 0 or actual.size == 0:
        raise ValueError("Пустые данные")

    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("Введены NAN или бесконечные значения")

    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Эпсилон должен быть положительным")

    # вычисление границ интервалов
    quantiles = np.linspace(0, 1, bins + 1)
    bin_edges = np.unique(np.quantile(expected, quantiles))

    if len(bin_edges) == 1:
        bin_edges = np.array([-np.inf, bin_edges[0], np.inf])
    else:
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

    expected_counts, expect_bins = np.histogram(expected, bins=bin_edges)
    actual_counts, actual_bins = np.histogram(actual, bins=bin_edges)

    expected_pct = (expected_counts + epsilon) / (
        expected_counts.sum() + epsilon * len(expected_counts)
    )
    actual_pct = (actual_counts + epsilon) / (
        actual_counts.sum() + epsilon * len(actual_counts)
    )

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

    return float(psi)
