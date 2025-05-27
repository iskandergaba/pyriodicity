import numpy as np
import pytest

from pyriodicity import SAZED


def test_sazed_data_nan():
    data = [np.nan] * 100
    with pytest.raises(ValueError):
        SAZED.detect(data)


def test_sazed_data_inf():
    data = [np.inf] * 100
    with pytest.raises(ValueError):
        SAZED.detect(data)


def test_sazed_data_const():
    data = [1] * 100
    period = SAZED.detect(data)
    assert period is None


def test_sazed_method_invalid(sinewave_10):
    data = sinewave_10
    with pytest.raises(ValueError):
        SAZED.detect(data, method="invalid")


def test_sinewave_10_sazed_majority(sinewave_10):
    data = sinewave_10
    period = SAZED.detect(data, method="majority")
    assert period is not None
    assert period == pytest.approx(10, abs=1)


def test_sinewave_50_sazed_majority(sinewave_50):
    data = sinewave_50
    period = SAZED.detect(data, method="majority")
    assert period is not None
    assert period == pytest.approx(50, abs=1)


def test_sinewave_100_sazed_majority(sinewave_100):
    data = sinewave_100
    period = SAZED.detect(data, method="majority")
    assert period is not None
    assert period == pytest.approx(100, abs=1)


def test_trianglewave_10_sazed_optimal(trianglewave_10):
    data = trianglewave_10
    period = SAZED.detect(data, method="optimal")
    assert period is not None
    assert period == pytest.approx(10, abs=1)


def test_trianglewave_50_sazed_optimal(trianglewave_50):
    data = trianglewave_50
    period = SAZED.detect(data, method="optimal")
    assert period is not None
    assert period == pytest.approx(50, abs=1)


def test_trianglewave_100_sazed_optimal(trianglewave_100):
    data = trianglewave_100
    period = SAZED.detect(data, method="optimal")
    assert period is not None
    assert period == pytest.approx(100, abs=1)


def test_co2_weekly_sazed_optimal(co2_weekly):
    data = co2_weekly
    period = SAZED.detect(data, method="optimal")
    assert period is not None
    assert period == pytest.approx(52, abs=1)


def test_co2_monthly_sazed_optimal(co2_monthly):
    data = co2_monthly
    period = SAZED.detect(data, method="optimal")
    assert period is not None
    assert period == pytest.approx(12, abs=1)


def test_co2_weekly_sazed_majority(co2_weekly):
    data = co2_weekly
    period = SAZED.detect(data, method="majority")
    assert period is not None
    assert period == pytest.approx(52, abs=1)


def test_co2_monthly_sazed_majority(co2_monthly):
    data = co2_monthly
    period = SAZED.detect(data, method="majority")
    assert period is not None
    assert period == pytest.approx(12, abs=1)


def test_co2_weekly_sazed_optimal_window_func_blackman(co2_weekly):
    data = co2_weekly
    period = SAZED.detect(data, method="optimal", window_func="blackman")
    assert period is not None
    assert period == pytest.approx(52, abs=1)


def test_co2_monthly_sazed_optimal_window_func_blackman(co2_monthly):
    data = co2_monthly
    period = SAZED.detect(data, method="optimal", window_func="blackman")
    assert period is not None
    assert period == pytest.approx(12, abs=1)


def test_co2_weekly_sazed_majority_window_func_hann(co2_weekly):
    data = co2_weekly
    period = SAZED.detect(data, method="majority", window_func="hann")
    assert period is not None
    assert period == pytest.approx(52, abs=1)


def test_co2_monthly_sazed_majority_window_func_hann(co2_monthly):
    data = co2_monthly
    period = SAZED.detect(data, method="majority", window_func="hann")
    assert period is not None
    assert period == pytest.approx(12, abs=1)
