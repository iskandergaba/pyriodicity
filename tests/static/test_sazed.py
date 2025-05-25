"""
Test SAZED (Spectral Autocorrelation Zero Ensemble Detector).

References
----------
.. [1] Gaba, Iskandar. (2023) SazedR: A package for estimating the season length
   of a seasonal time series.
   https://github.com/cran/sazedR
"""

import numpy as np
import pytest

from pyriodicity import SAZED


def test_sinewave_10_sazed_optimal(sinewave_10):
    """Test SAZED optimal method on sine wave with period 10."""
    data = sinewave_10
    period = SAZED.detect(data, method="optimal")
    assert period is not None
    assert period == pytest.approx(10, abs=1)


def test_sinewave_50_sazed_optimal(sinewave_50):
    """Test SAZED optimal method on sine wave with period 50."""
    data = sinewave_50
    period = SAZED.detect(data, method="optimal")
    assert period is not None
    assert period == pytest.approx(50, abs=1)


def test_sinewave_100_sazed_optimal(sinewave_100):
    """Test SAZED optimal method on sine wave with period 100."""
    data = sinewave_100
    period = SAZED.detect(data, method="optimal")
    assert period is not None
    assert period == pytest.approx(100, abs=1)


def test_trianglewave_10_sazed_optimal(trianglewave_10):
    """Test SAZED optimal method on triangle wave with period 10."""
    data = trianglewave_10
    period = SAZED.detect(data, method="optimal")
    assert period is not None
    assert period == pytest.approx(10, abs=1)


def test_trianglewave_50_sazed_optimal(trianglewave_50):
    """Test SAZED optimal method on triangle wave with period 50."""
    data = trianglewave_50
    period = SAZED.detect(data, method="optimal")
    assert period is not None
    assert period == pytest.approx(50, abs=1)


def test_trianglewave_100_sazed_optimal(trianglewave_100):
    """Test SAZED optimal method on triangle wave with period 100."""
    data = trianglewave_100
    period = SAZED.detect(data, method="optimal")
    assert period is not None
    assert period == pytest.approx(100, abs=1)


def test_co2_weekly_sazed_optimal(co2_weekly):
    """Test SAZED optimal method on weekly CO2 data."""
    data = co2_weekly
    period = SAZED.detect(data, method="optimal")
    assert period is not None
    assert period == pytest.approx(52, abs=1)


def test_co2_monthly_sazed_optimal(co2_monthly):
    """Test SAZED optimal method on monthly CO2 data."""
    data = co2_monthly
    period = SAZED.detect(data, method="optimal")
    assert period is not None
    assert period == pytest.approx(12, abs=1)


def test_co2_weekly_sazed_majority(co2_weekly):
    """Test SAZED majority method on weekly CO2 data."""
    data = co2_weekly
    period = SAZED.detect(data, method="majority")
    assert period is not None
    assert period == pytest.approx(52, abs=1)


def test_co2_monthly_sazed_majority(co2_monthly):
    """Test SAZED majority method on monthly CO2 data."""
    data = co2_monthly
    period = SAZED.detect(data, method="majority")
    assert period is not None
    assert period == pytest.approx(12, abs=1)


def test_co2_weekly_sazed_optimal_window_func_blackman(co2_weekly):
    """Test SAZED optimal method with Blackman window on weekly CO2 data."""
    data = co2_weekly
    period = SAZED.detect(data, method="optimal", window_func="blackman")
    assert period is not None
    assert period == pytest.approx(52, abs=1)


def test_co2_monthly_sazed_optimal_window_func_blackman(co2_monthly):
    """Test SAZED optimal method with Blackman window on monthly CO2 data."""
    data = co2_monthly
    period = SAZED.detect(data, method="optimal", window_func="blackman")
    assert period is not None
    assert period == pytest.approx(12, abs=1)


def test_sazed_invalid_input():
    """Test SAZED behavior with invalid input."""
    # Test with constant input
    data = [1] * 100
    period = SAZED.detect(data)
    assert period is None

    # Test with NaN
    data = [np.nan] * 100
    period = SAZED.detect(data)
    assert period is None

    # Test with infinity
    data = [np.inf] * 100
    period = SAZED.detect(data)
    assert period is None

    # Test with complex numbers
    data = [1 + 1j] * 100
    period = SAZED.detect(data)
    assert period is None

    # Test with too short data
    data = [1, 2, 3]
    period = SAZED.detect(data)
    assert period is None

    # Test with zero variance data
    data = np.zeros(100)
    period = SAZED.detect(data)
    assert period is None


def test_sazed_method_validation():
    """Test SAZED method parameter validation."""
    data = np.sin(2 * np.pi * np.arange(100) / 10)

    # Test invalid method
    with pytest.raises(ValueError):
        SAZED.detect(data, method="invalid")
