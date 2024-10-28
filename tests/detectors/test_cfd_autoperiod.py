from pyriodicity import CFDAutoperiod


def test_co2_daily_cfd_autoperiod_default(co2_daily):
    data = co2_daily
    periods = CFDAutoperiod.detect(data)
    assert len(periods) > 0
    assert 364 in periods


def test_co2_weekly_cfd_autoperiod_default(co2_weekly):
    data = co2_weekly
    periods = CFDAutoperiod.detect(data)
    assert len(periods) > 0
    assert 52 in periods


def test_co2_monthly_cfd_autoperiod_default(co2_monthly):
    data = co2_monthly
    periods = CFDAutoperiod.detect(data)
    assert len(periods) > 0
    assert 12 in periods


def test_co2_daily_cfd_autoperiod_detrend_func_none(co2_daily):
    data = co2_daily
    periods = CFDAutoperiod.detect(data, detrend_func=None)
    assert len(periods) == 0


def test_co2_weekly_cfd_autoperiod_detrend_func_none(co2_weekly):
    data = co2_weekly
    periods = CFDAutoperiod.detect(data, detrend_func=None)
    assert len(periods) == 0


def test_co2_monthly_cfd_autoperiod_detrend_func_none(co2_monthly):
    data = co2_monthly
    periods = CFDAutoperiod.detect(data, detrend_func=None)
    assert len(periods) == 0


def test_co2_daily_cfd_autoperiod_detrend_func_constant(co2_daily):
    data = co2_daily
    periods = CFDAutoperiod.detect(data, detrend_func="constant")
    assert len(periods) == 0


def test_co2_weekly_cfd_autoperiod_detrend_func_constant(co2_weekly):
    data = co2_weekly
    periods = CFDAutoperiod.detect(data, detrend_func="constant")
    assert len(periods) == 0


def test_co2_monthly_cfd_autoperiod_detrend_func_constant(co2_monthly):
    data = co2_monthly
    periods = CFDAutoperiod.detect(data, detrend_func="constant")
    assert len(periods) == 0


def test_co2_daily_cfd_autoperiod_window_func_blackman(co2_daily):
    data = co2_daily
    periods = CFDAutoperiod.detect(data, window_func="blackman")
    assert len(periods) > 0
    assert 364 in periods


def test_co2_weekly_cfd_autoperiod_window_func_blackman(co2_weekly):
    data = co2_weekly
    periods = CFDAutoperiod.detect(data, window_func="blackman")
    assert len(periods) > 0
    assert 52 in periods


def test_co2_monthly_cfd_autoperiod_window_func_blackman(co2_monthly):
    data = co2_monthly
    periods = CFDAutoperiod.detect(data, window_func="blackman")
    assert len(periods) > 0
    assert 12 in periods
