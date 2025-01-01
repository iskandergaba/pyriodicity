from pyriodicity import CFDAutoperiod


def test_sinewave_10_cfd_autoperiod_find_all_periods(sinewave_10):
    data = sinewave_10
    periods = CFDAutoperiod.detect(data)
    assert len(periods) > 0
    assert 10 in periods


def test_sinewave_50_cfd_autoperiod_find_all_periods(sinewave_50):
    data = sinewave_50
    periods = CFDAutoperiod.detect(data)
    assert len(periods) > 0
    assert 50 in periods


def test_sinewave_100_cfd_autoperiod_find_all_periods(sinewave_100):
    data = sinewave_100
    periods = CFDAutoperiod.detect(data)
    assert len(periods) > 0
    assert 100 in periods


def test_trianglewave_10_cfd_autoperiod_find_all_periods(trianglewave_10):
    data = trianglewave_10
    periods = CFDAutoperiod.detect(data)
    assert len(periods) > 0
    assert 10 in periods


def test_trianglewave_50_cfd_autoperiod_find_all_periods(trianglewave_50):
    data = trianglewave_50
    periods = CFDAutoperiod.detect(data)
    assert len(periods) > 0
    assert 50 in periods


def test_trianglewave_100_cfd_autoperiod_find_all_periods(trianglewave_100):
    data = trianglewave_100
    periods = CFDAutoperiod.detect(data)
    assert len(periods) > 0
    assert 100 in periods


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


def test_co2_weekly_cfd_autoperiod_detrend_func_none(co2_weekly):
    data = co2_weekly
    periods = CFDAutoperiod.detect(data, detrend_func=None)
    assert len(periods) == 0


def test_co2_monthly_cfd_autoperiod_detrend_func_none(co2_monthly):
    data = co2_monthly
    periods = CFDAutoperiod.detect(data, detrend_func=None)
    assert len(periods) == 0


def test_co2_weekly_cfd_autoperiod_detrend_func_constant(co2_weekly):
    data = co2_weekly
    periods = CFDAutoperiod.detect(data, detrend_func="constant")
    assert len(periods) == 0


def test_co2_monthly_cfd_autoperiod_detrend_func_constant(co2_monthly):
    data = co2_monthly
    periods = CFDAutoperiod.detect(data, detrend_func="constant")
    assert len(periods) == 0


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
