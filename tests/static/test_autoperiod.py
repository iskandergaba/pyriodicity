from pyriodicity import Autoperiod


def test_sinewave_10_autoperiod_find_all_periods(sinewave_10):
    data = sinewave_10
    periods = Autoperiod.detect(data)
    assert len(periods) > 0
    assert 10 in periods


def test_sinewave_50_autoperiod_find_all_periods(sinewave_50):
    data = sinewave_50
    periods = Autoperiod.detect(data)
    assert len(periods) > 0
    assert 50 in periods


def test_sinewave_100_autoperiod_find_all_periods(sinewave_100):
    data = sinewave_100
    periods = Autoperiod.detect(data)
    assert len(periods) > 0
    assert 100 in periods


def test_trianglewave_10_autoperiod_find_all_periods(trianglewave_10):
    data = trianglewave_10
    periods = Autoperiod.detect(data)
    assert len(periods) > 0
    assert 10 in periods


def test_trianglewave_50_autoperiod_find_all_periods(trianglewave_50):
    data = trianglewave_50
    periods = Autoperiod.detect(data)
    assert len(periods) > 0
    assert 50 in periods


def test_trianglewave_100_autoperiod_find_all_periods(trianglewave_100):
    data = trianglewave_100
    periods = Autoperiod.detect(data)
    assert len(periods) > 0
    assert 100 in periods


def test_co2_weekly_autoperiod_default(co2_weekly):
    data = co2_weekly
    periods = Autoperiod.detect(data)
    assert len(periods) > 0
    assert 52 in periods


def test_co2_monthly_autoperiod_default(co2_monthly):
    data = co2_monthly
    periods = Autoperiod.detect(data)
    assert len(periods) > 0
    assert 12 in periods


def test_co2_weekly_autoperiod_detrend_func_none(co2_weekly):
    data = co2_weekly
    periods = Autoperiod.detect(data, detrend_func=None)
    assert len(periods) == 0


def test_co2_monthly_autoperiod_detrend_func_none(co2_monthly):
    data = co2_monthly
    periods = Autoperiod.detect(data, detrend_func=None)
    assert len(periods) == 0


def test_co2_weekly_autoperiod_detrend_func_constant(co2_weekly):
    data = co2_weekly
    periods = Autoperiod.detect(data, detrend_func="constant")
    assert len(periods) == 0


def test_co2_monthly_autoperiod_detrend_func_constant(co2_monthly):
    data = co2_monthly
    periods = Autoperiod.detect(data, detrend_func="constant")
    assert len(periods) == 0


def test_co2_weekly_autoperiod_detrend_func_constant_window_func_blackman(co2_weekly):
    data = co2_weekly
    periods = Autoperiod.detect(data, detrend_func="constant", window_func="blackman")
    assert len(periods) > 0
    assert 52 in periods


def test_co2_monthly_autoperiod_detrend_func_constant_window_func_blackman(co2_monthly):
    data = co2_monthly
    periods = Autoperiod.detect(data, detrend_func="constant", window_func="blackman")
    assert len(periods) > 0
    assert 12 in periods
