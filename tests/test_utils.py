import numpy as np
import pytest
from src.utils import correlation_coefficient, rmse, pbias, nse


def test_correlation_coefficient_perfect():
    a = np.array([1, 2, 3, 4])
    assert correlation_coefficient(a, a) == pytest.approx(1.0)


def test_rmse_zero():
    a = np.array([1, 2, 3])
    assert rmse(a, a) == pytest.approx(0.0)


def test_pbias_zero():
    a = np.array([1, 2, 3])
    assert pbias(a, a) == pytest.approx(0.0)


def test_pbias_zero_denominator():
    obs = np.zeros(3)
    pred = np.array([1, 2, 3])
    assert np.isnan(pbias(obs, pred))


def test_nse_perfect():
    a = np.array([1, 2, 3])
    assert nse(a, a) == pytest.approx(1.0)


def test_nse_zero_denominator():
    obs = np.array([5, 5, 5])
    pred = np.array([4, 6, 5])
    # denominator zero for nse
    assert np.isnan(nse(obs, pred))
