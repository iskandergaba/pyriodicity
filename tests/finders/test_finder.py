from statsmodels.datasets import co2

from auto_period_finder import AutoPeriodFinder
from auto_period_finder.enums import TimeSeriesDecomposer


def test_find_all_periods():
    data = co2.load().data.resample("ME").mean().ffill()
    period_finder = AutoPeriodFinder(data)
    periods = period_finder.fit()
    assert len(periods) != 0


def test_find_first_two_periods():
    data = co2.load().data.resample("ME").mean().ffill()
    period_finder = AutoPeriodFinder(data)
    periods = period_finder.fit(max_period_count=2)
    assert len(periods) == 2


def test_find_strongest_period_acf_wise():
    data = co2.load().data.resample("ME").mean().ffill()
    period_finder = AutoPeriodFinder(data)
    periods = period_finder.fit(max_period_count=1)
    strongest_period_acf = period_finder.fit_find_strongest_acf()
    assert len(periods) == 1
    assert periods[0] == strongest_period_acf


def test_find_strongest_period_var_wise_stl_default():
    data = co2.load().data.resample("ME").mean().ffill()
    period_finder = AutoPeriodFinder(data)
    strongest_period_var = period_finder.fit_find_strongest_var(
        decomposer=TimeSeriesDecomposer.STL
    )
    assert strongest_period_var == 180


def test_find_strongest_period_var_wise_stl_custom():
    data = co2.load().data.resample("ME").mean().ffill()
    period_finder = AutoPeriodFinder(data)
    strongest_period_var = period_finder.fit_find_strongest_var(
        decomposer=TimeSeriesDecomposer.STL, decomposer_kwargs={"seasonal_deg": 0}
    )
    assert strongest_period_var == 180


def test_find_strongest_period_var_wise_moving_averages():
    data = co2.load().data.resample("ME").mean().ffill()
    period_finder = AutoPeriodFinder(data)
    strongest_period_var = period_finder.fit_find_strongest_var()
    assert strongest_period_var == 180
