from statsmodels.datasets import co2

from auto_period_finder import AutocorrelationPeriodFinder


def test_co2_daily_acf_default():
    data = co2.load().data.resample("D").mean().ffill()
    period_finder = AutocorrelationPeriodFinder(data)
    periods = period_finder.fit()
    assert len(periods) > 0


def test_co2_weekly_acf_default():
    data = co2.load().data.resample("W").mean().ffill()
    period_finder = AutocorrelationPeriodFinder(data)
    periods = period_finder.fit()
    assert len(periods) > 0


def test_co2_monthly_acf_default():
    data = co2.load().data.resample("ME").mean().ffill()
    period_finder = AutocorrelationPeriodFinder(data)
    periods = period_finder.fit()
    assert len(periods) > 0


def test_co2_daily_acf_max_period_count_one():
    data = co2.load().data.resample("D").mean().ffill()
    period_finder = AutocorrelationPeriodFinder(data)
    periods = period_finder.fit(max_period_count=1)
    assert len(periods) == 1
    assert periods[0] == 364


def test_co2_weekly_acf_max_period_count_one():
    data = co2.load().data.resample("W").mean().ffill()
    period_finder = AutocorrelationPeriodFinder(data)
    periods = period_finder.fit(max_period_count=1)
    assert len(periods) == 1
    assert periods[0] == 52


def test_co2_monthly_acf_max_period_count_one():
    data = co2.load().data.resample("ME").mean().ffill()
    period_finder = AutocorrelationPeriodFinder(data)
    periods = period_finder.fit(max_period_count=1)
    assert len(periods) == 1
    assert periods[0] == 12


def test_co2_daily_acf_max_period_count_one_correlation_func_spearman():
    data = co2.load().data.resample("D").mean().ffill()
    period_finder = AutocorrelationPeriodFinder(data)
    periods = period_finder.fit(max_period_count=1, correlation_func="spearman")
    assert len(periods) == 1
    assert periods[0] == 364


def test_co2_weekly_acf_max_period_count_one_correlation_func_spearman():
    data = co2.load().data.resample("W").mean().ffill()
    period_finder = AutocorrelationPeriodFinder(data)
    periods = period_finder.fit(max_period_count=1, correlation_func="spearman")
    assert len(periods) == 1
    assert periods[0] == 52


def test_co2_monthly_acf_max_period_count_one_correlation_func_spearman():
    data = co2.load().data.resample("ME").mean().ffill()
    period_finder = AutocorrelationPeriodFinder(data)
    periods = period_finder.fit(max_period_count=1, correlation_func="spearman")
    assert len(periods) == 1
    assert periods[0] == 12


def test_co2_daily_acf_max_period_count_one_window_func_blackman():
    data = co2.load().data.resample("D").mean().ffill()
    period_finder = AutocorrelationPeriodFinder(data)
    periods = period_finder.fit(max_period_count=1, window_func="blackman")
    assert len(periods) == 1
    assert periods[0] == 364


def test_co2_weekly_acf_max_period_count_one_window_func_blackman():
    data = co2.load().data.resample("W").mean().ffill()
    period_finder = AutocorrelationPeriodFinder(data)
    periods = period_finder.fit(max_period_count=1, window_func="blackman")
    assert len(periods) == 1
    assert periods[0] == 52


def test_co2_monthly_acf_max_period_count_one_window_func_blackman():
    data = co2.load().data.resample("ME").mean().ffill()
    period_finder = AutocorrelationPeriodFinder(data)
    periods = period_finder.fit(max_period_count=1, window_func="blackman")
    assert len(periods) == 1
    assert periods[0] == 12
