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
    assert any(period == pytest.approx(52, abs=1) for period in periods)


def test_co2_monthly_robustperiod_default(co2_monthly):
    data = co2_monthly
    periods = RobustPeriod.detect(data)
    assert len(periods) > 0
    assert any(period == pytest.approx(12, abs=1) for period in periods)


def test_co2_weekly_robustperiod_max_period_count_one(co2_weekly):
    data = co2_weekly
    periods = RobustPeriod.detect(data, max_period_count=1)
    assert len(periods) == 1


def test_co2_monthly_robustperiod_max_period_count_one(co2_monthly):
    data = co2_monthly
    periods = RobustPeriod.detect(data, max_period_count=1)
    assert len(periods) == 1


def test_co2_weekly_robustperiod_c_1_db_n_4_modwt_level_8(co2_weekly):
    data = co2_weekly
    periods = RobustPeriod.detect(data, c=1, db_n=4, modwt_level=8)
    assert len(periods) > 0
    assert any(period == pytest.approx(52, abs=1) for period in periods)


def test_co2_monthly_robustperiod_c_1_db_n_4_modwt_level_8(co2_monthly):
    data = co2_monthly
    periods = RobustPeriod.detect(data, c=1, db_n=4, modwt_level=8)
    assert len(periods) > 0
    assert any(period == pytest.approx(12, abs=1) for period in periods)


def test_co2_weekly_robustperiod_lamb_hodrick_prescott_c_1_delta_1(co2_weekly):
    data = co2_weekly
    periods = RobustPeriod.detect(data, lamb="hodrick-prescott", c=1, delta=1)
    assert len(periods) > 0
    assert any(period == pytest.approx(52, abs=1) for period in periods)


def test_co2_monthly_robustperiod_lamb_hodrick_prescott_c_1_delta_1(co2_monthly):
    data = co2_monthly
    periods = RobustPeriod.detect(data, lamb="hodrick-prescott", c=1, delta=1)
    assert len(periods) > 0
    assert any(period == pytest.approx(12, abs=1) for period in periods)


def test_co2_weekly_robustperiod_lamb_1e6_modwt_level_12(co2_weekly):
    data = co2_weekly
    periods = RobustPeriod.detect(data, lamb=1e6, modwt_level=12)
    assert len(periods) > 0
    assert any(period == pytest.approx(52, abs=1) for period in periods)


def test_co2_monthly_robustperiod_lamb_1e6_modwt_level_12_lamb_1e6(co2_monthly):
    data = co2_monthly
    periods = RobustPeriod.detect(data, lamb=1e6, modwt_level=12)
    assert len(periods) > 0
    assert any(period == pytest.approx(12, abs=1) for period in periods)
