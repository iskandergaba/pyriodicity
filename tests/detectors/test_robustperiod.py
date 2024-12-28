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


def test_co2_weekly_robustperiod_max_period_count_one(co2_weekly):
    data = co2_weekly
    periods = RobustPeriod.detect(data, max_period_count=1)
    assert len(periods) == 1


def test_co2_monthly_robustperiod_max_period_count_one(co2_monthly):
    data = co2_monthly
    periods = RobustPeriod.detect(data, max_period_count=1)
    assert len(periods) == 1
