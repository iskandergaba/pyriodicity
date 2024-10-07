from statsmodels.datasets import co2

from pyriodicity import Autoperiod


def test_co2_daily_autoperiod_default():
    data = co2.load().data.resample("D").mean().ffill()
    autoperiod = Autoperiod(data)
    periods = autoperiod.fit()
    assert len(periods) == 1
    assert 364 in periods


def test_co2_weekly_autoperiod_default():
    data = co2.load().data.resample("W").mean().ffill()
    autoperiod = Autoperiod(data)
    periods = autoperiod.fit()
    assert len(periods) == 1
    assert 52 in periods


def test_co2_monthly_autoperiod_default():
    data = co2.load().data.resample("ME").mean().ffill()
    autoperiod = Autoperiod(data)
    periods = autoperiod.fit()
    assert len(periods) == 1
    assert 12 in periods


def test_co2_daily_autoperiod_detrend_func_constant():
    data = co2.load().data.resample("D").mean().ffill()
    autoperiod = Autoperiod(data)
    periods = autoperiod.fit(detrend_func="constant")
    assert len(periods) == 0


def test_co2_weekly_autoperiod_detrend_func_constant():
    data = co2.load().data.resample("W").mean().ffill()
    autoperiod = Autoperiod(data)
    periods = autoperiod.fit(detrend_func="constant")
    assert len(periods) == 0


def test_co2_monthly_autoperiod_detrend_func_constant():
    data = co2.load().data.resample("ME").mean().ffill()
    autoperiod = Autoperiod(data)
    periods = autoperiod.fit(detrend_func="constant")
    assert len(periods) == 0


def test_co2_daily_autoperiod_detrend_func_constant_window_func_blackman():
    data = co2.load().data.resample("D").mean().ffill()
    autoperiod = Autoperiod(data)
    periods = autoperiod.fit(detrend_func="constant", window_func="blackman")
    assert len(periods) == 1
    assert 364 in periods


def test_co2_weekly_autoperiod_detrend_func_constant_window_func_blackman():
    data = co2.load().data.resample("W").mean().ffill()
    autoperiod = Autoperiod(data)
    periods = autoperiod.fit(detrend_func="constant", window_func="blackman")
    assert len(periods) == 1
    assert 52 in periods


def test_co2_monthly_autoperiod_detrend_func_constant_window_func_blackman():
    data = co2.load().data.resample("ME").mean().ffill()
    autoperiod = Autoperiod(data)
    periods = autoperiod.fit(detrend_func="constant", window_func="blackman")
    assert len(periods) == 1
    assert 12 in periods
