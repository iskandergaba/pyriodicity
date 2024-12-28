from pyriodicity import RobustPeriod


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
