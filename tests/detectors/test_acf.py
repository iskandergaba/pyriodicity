from pyriodicity import ACFPeriodicityDetector


def test_co2_daily_acf_default(co2_daily):
    data = co2_daily
    acf_detector = ACFPeriodicityDetector(data)
    periods = acf_detector.fit()
    assert len(periods) > 0


def test_co2_weekly_acf_default(co2_weekly):
    data = co2_weekly
    acf_detector = ACFPeriodicityDetector(data)
    periods = acf_detector.fit()
    assert len(periods) > 0


def test_co2_monthly_acf_default(co2_monthly):
    data = co2_monthly
    acf_detector = ACFPeriodicityDetector(data)
    periods = acf_detector.fit()
    assert len(periods) > 0


def test_co2_daily_acf_max_period_count_one(co2_daily):
    data = co2_daily
    acf_detector = ACFPeriodicityDetector(data)
    periods = acf_detector.fit(max_period_count=1)
    assert len(periods) == 1
    assert 364 in periods


def test_co2_weekly_acf_max_period_count_one(co2_weekly):
    data = co2_weekly
    acf_detector = ACFPeriodicityDetector(data)
    periods = acf_detector.fit(max_period_count=1)
    assert len(periods) == 1
    assert 52 in periods


def test_co2_monthly_acf_max_period_count_one(co2_monthly):
    data = co2_monthly
    acf_detector = ACFPeriodicityDetector(data)
    periods = acf_detector.fit(max_period_count=1)
    assert len(periods) == 1
    assert 12 in periods


def test_co2_daily_acf_max_period_count_one_correlation_func_spearman(co2_daily):
    data = co2_daily
    acf_detector = ACFPeriodicityDetector(data)
    periods = acf_detector.fit(max_period_count=1, correlation_func="spearman")
    assert len(periods) == 1
    assert 364 in periods


def test_co2_weekly_acf_max_period_count_one_correlation_func_spearman(co2_weekly):
    data = co2_weekly
    acf_detector = ACFPeriodicityDetector(data)
    periods = acf_detector.fit(max_period_count=1, correlation_func="spearman")
    assert len(periods) == 1
    assert 52 in periods


def test_co2_monthly_acf_max_period_count_one_correlation_func_spearman(co2_monthly):
    data = co2_monthly
    acf_detector = ACFPeriodicityDetector(data)
    periods = acf_detector.fit(max_period_count=1, correlation_func="spearman")
    assert len(periods) == 1
    assert 12 in periods


def test_co2_daily_acf_max_period_count_one_correlation_func_kendall(co2_daily):
    data = co2_daily
    acf_detector = ACFPeriodicityDetector(data)
    periods = acf_detector.fit(max_period_count=1, correlation_func="kendall")
    assert len(periods) == 1
    assert 364 in periods


def test_co2_weekly_acf_max_period_count_one_correlation_func_kendall(co2_weekly):
    data = co2_weekly
    acf_detector = ACFPeriodicityDetector(data)
    periods = acf_detector.fit(max_period_count=1, correlation_func="kendall")
    assert len(periods) == 1
    assert 52 in periods


def test_co2_monthly_acf_max_period_count_one_correlation_func_kendall(co2_monthly):
    data = co2_monthly
    acf_detector = ACFPeriodicityDetector(data)
    periods = acf_detector.fit(max_period_count=1, correlation_func="kendall")
    assert len(periods) == 1
    assert 12 in periods


def test_co2_daily_acf_max_period_count_one_window_func_blackman(co2_daily):
    data = co2_daily
    acf_detector = ACFPeriodicityDetector(data)
    periods = acf_detector.fit(max_period_count=1, window_func="blackman")
    assert len(periods) == 1
    assert 364 in periods


def test_co2_weekly_acf_max_period_count_one_window_func_blackman(co2_weekly):
    data = co2_weekly
    acf_detector = ACFPeriodicityDetector(data)
    periods = acf_detector.fit(max_period_count=1, window_func="blackman")
    assert len(periods) == 1
    assert 52 in periods


def test_co2_monthly_acf_max_period_count_one_window_func_blackman(co2_monthly):
    data = co2_monthly
    acf_detector = ACFPeriodicityDetector(data)
    periods = acf_detector.fit(max_period_count=1, window_func="blackman")
    assert len(periods) == 1
    assert 12 in periods
