import pytest

from pyriodicity import RobustPeriod


def test_sinewave_10_robustperiod_default(sinewave_10):
    data = sinewave_10
    with pytest.raises(AttributeError):
        RobustPeriod.detect(data)


def test_co2_weekly_index_reset_robustperiod_default(co2_weekly):
    data = co2_weekly.reset_index(drop=True)
    with pytest.raises(TypeError):
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
