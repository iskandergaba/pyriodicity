import numpy as np
from statsmodels.datasets import co2

from auto_period_finder import AutoperiodDetector


def test_co2_daily_autoperiod_default():
    data = co2.load().data.resample("D").mean().ffill()
    autoperiod_detector = AutoperiodDetector(data)
    periods = autoperiod_detector.fit()
    assert len(periods) == 1
    assert periods[0] == 364


def test_co2_weekly_autoperiod_default():
    data = co2.load().data.resample("W").mean().ffill()
    autoperiod_detector = AutoperiodDetector(data)
    periods = autoperiod_detector.fit()
    assert len(periods) == 1
    assert periods[0] == 52


def test_co2_monthly_autoperiod_default():
    data = co2.load().data.resample("ME").mean().ffill()
    autoperiod_detector = AutoperiodDetector(data)
    periods = autoperiod_detector.fit()
    assert len(periods) == 1
    assert periods[0] == 12


def test_co2_daily_autoperiod_detrend_func_constant():
    data = co2.load().data.resample("D").mean().ffill()
    autoperiod_detector = AutoperiodDetector(data)
    periods = autoperiod_detector.fit(detrend_func="constant")
    assert len(periods) == 0


def test_co2_weekly_autoperiod_detrend_func_constant():
    data = co2.load().data.resample("W").mean().ffill()
    autoperiod_detector = AutoperiodDetector(data)
    periods = autoperiod_detector.fit(detrend_func="constant")
    assert len(periods) == 0


def test_co2_monthly_autoperiod_detrend_func_constant():
    data = co2.load().data.resample("ME").mean().ffill()
    autoperiod_detector = AutoperiodDetector(data)
    periods = autoperiod_detector.fit(detrend_func="constant")
    assert len(periods) == 0


def test_co2_daily_autoperiod_detrend_func_constant_window_func_blackman():
    data = co2.load().data.resample("D").mean().ffill()
    autoperiod_detector = AutoperiodDetector(data)
    periods = autoperiod_detector.fit(detrend_func="constant", window_func="blackman")
    assert len(periods) == 1
    assert periods[0] == 364


def test_co2_weekly_autoperiod_detrend_func_constant_window_func_blackman():
    data = co2.load().data.resample("ME").mean().ffill()
    autoperiod_detector = AutoperiodDetector(data)
    periods = autoperiod_detector.fit(detrend_func="constant", window_func="blackman")
    assert len(periods) == 1
    assert periods[0] == 12


def test_co2_monthly_autoperiod_detrend_func_constant_window_func_blackman():
    data = co2.load().data.resample("ME").mean().ffill()
    autoperiod_detector = AutoperiodDetector(data)
    periods = autoperiod_detector.fit(detrend_func="constant", window_func="blackman")
    assert len(periods) == 1
    assert periods[0] == 12


def test_co2_daily_autoperiod_detrend_func_custom_window_func_blackmanharris():
    data = co2.load().data.resample("D").mean().ffill()
    autoperiod_detector = AutoperiodDetector(data)
    periods = autoperiod_detector.fit(
        detrend_func=lambda x: x - np.median(x), window_func="blackmanharris"
    )
    assert len(periods) == 1
    assert periods[0] == 364


def test_co2_weekly_autoperiod_detrend_func_custom_window_func_blackmanharris():
    data = co2.load().data.resample("W").mean().ffill()
    autoperiod_detector = AutoperiodDetector(data)
    periods = autoperiod_detector.fit(
        detrend_func=lambda x: x - np.median(x), window_func="blackmanharris"
    )
    assert len(periods) == 1
    assert periods[0] == 52


def test_co2_monthly_autoperiod_detrend_func_custom_window_func_blackmanharris():
    data = co2.load().data.resample("ME").mean().ffill()
    autoperiod_detector = AutoperiodDetector(data)
    periods = autoperiod_detector.fit(
        detrend_func=lambda x: x - np.median(x), window_func="blackmanharris"
    )
    assert len(periods) == 1
    assert periods[0] == 12
