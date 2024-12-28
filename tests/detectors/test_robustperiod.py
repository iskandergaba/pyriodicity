import pytest

from pyriodicity import RobustPeriod


def test_sinewave_10_robustperiod_default(sinewave_10):
    data = sinewave_10
    with pytest.raises(AttributeError):
        RobustPeriod.detect(data)


def test_sinewave_50_robustperiod_default(sinewave_50):
    data = sinewave_50
    with pytest.raises(AttributeError):
        RobustPeriod.detect(data)


def test_sinewave_100_robustperiod_default(sinewave_100):
    data = sinewave_100
    with pytest.raises(AttributeError):
        RobustPeriod.detect(data)


def test_trianglewave_10_robustperiod_default(trianglewave_10):
    data = trianglewave_10
    with pytest.raises(AttributeError):
        RobustPeriod.detect(data)


def test_trianglewave_50_robustperiod_default(trianglewave_50):
    data = trianglewave_50
    with pytest.raises(AttributeError):
        RobustPeriod.detect(data)


def test_trianglewave_100_robustperiod_default(trianglewave_100):
    data = trianglewave_100
    with pytest.raises(AttributeError):
        RobustPeriod.detect(data)


def test_co2_weekly_robustperiod_default(co2_weekly):
    data = co2_weekly
    periods = RobustPeriod.detect(data)
    assert len(periods) > 0
    assert 52 in periods


def test_co2_monthly_robustperiod_default(co2_monthly):
    data = co2_monthly
    periods = RobustPeriod.detect(data)
    assert len(periods) > 0
    assert 12 in periods


def test_co2_weekly_robustperiod_max_period_count_one(co2_weekly):
    data = co2_weekly
    periods = RobustPeriod.detect(data, max_period_count=1)
    assert len(periods) == 1


def test_co2_monthly_robustperiod_max_period_count_one(co2_monthly):
    data = co2_monthly
    periods = RobustPeriod.detect(data, max_period_count=1)
    assert len(periods) == 1


def test_co2_weekly_robustperiod_lamb_hodrick_prescott(co2_weekly):
    data = co2_weekly
    periods = RobustPeriod.detect(data, lamb="hodrick-prescott")
    assert len(periods) > 0
    assert 2 in periods


def test_co2_monthly_robustperiod_lamb_hodrick_prescott(co2_monthly):
    data = co2_monthly
    periods = RobustPeriod.detect(data, lamb="hodrick-prescott")
    assert len(periods) > 0
    assert 12 in periods
