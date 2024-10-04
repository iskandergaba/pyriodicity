from statsmodels.datasets import co2

from pyriodicity import CFDAutoperiod


def test_co2_daily_cfd_autoperiod_default():
    data = co2.load().data.resample("D").mean().ffill()
    autoperiod = CFDAutoperiod(data)
    periods = autoperiod.fit()
    assert len(periods) == 1
    assert periods[0] == 364


def test_co2_weekly_cfd_autoperiod_default():
    data = co2.load().data.resample("W").mean().ffill()
    autoperiod = CFDAutoperiod(data)
    periods = autoperiod.fit()
    assert len(periods) == 1
    assert periods[0] == 52


def test_co2_monthly_cfd_autoperiod_default():
    data = co2.load().data.resample("ME").mean().ffill()
    autoperiod = CFDAutoperiod(data)
    periods = autoperiod.fit()
    assert len(periods) == 1
    assert periods[0] == 12


def test_co2_daily_cfd_autoperiod_detrend_func_constant():
    data = co2.load().data.resample("D").mean().ffill()
    autoperiod = CFDAutoperiod(data)
    periods = autoperiod.fit(detrend_func="constant")
    assert len(periods) == 0


def test_co2_weekly_cfd_autoperiod_detrend_func_constant():
    data = co2.load().data.resample("W").mean().ffill()
    autoperiod = CFDAutoperiod(data)
    periods = autoperiod.fit(detrend_func="constant")
    assert len(periods) == 0


def test_co2_monthly_cfd_autoperiod_detrend_func_constant():
    data = co2.load().data.resample("ME").mean().ffill()
    autoperiod = CFDAutoperiod(data)
    periods = autoperiod.fit(detrend_func="constant")
    assert len(periods) == 0


def test_co2_daily_cfd_autoperiod_detrend_func_constant_window_func_blackman():
    data = co2.load().data.resample("D").mean().ffill()
    autoperiod = CFDAutoperiod(data)
    periods = autoperiod.fit(detrend_func="constant", window_func="blackman")
    assert len(periods) == 1
    assert periods[0] == 364


def test_co2_weekly_cfd_autoperiod_detrend_func_constant_window_func_blackman():
    data = co2.load().data.resample("ME").mean().ffill()
    autoperiod = CFDAutoperiod(data)
    periods = autoperiod.fit(detrend_func="constant", window_func="blackman")
    assert len(periods) == 1
    assert periods[0] == 12


def test_co2_monthly_cfd_autoperiod_detrend_func_constant_window_func_blackman():
    data = co2.load().data.resample("ME").mean().ffill()
    autoperiod = CFDAutoperiod(data)
    periods = autoperiod.fit(detrend_func="constant", window_func="blackman")
    assert len(periods) == 1
    assert periods[0] == 12
