from statsmodels.datasets import co2
from auto_period_finder import FourierPeriodFinder


def test_fourier_find_all_periods():
    data = co2.load().data.resample("ME").mean().ffill()
    period_finder = FourierPeriodFinder(data)
    periods = period_finder.fit()
    assert len(periods) != 0


def test_fourier_find_first_two_periods():
    data = co2.load().data.resample("ME").mean().ffill()
    period_finder = FourierPeriodFinder(data)
    periods = period_finder.fit(max_period_count=2)
    assert len(periods) == 2


def test_fourier_find_strongest_period_no_window_func():
    data = co2.load().data.resample("ME").mean().ffill()
    period_finder = FourierPeriodFinder(data)
    periods = period_finder.fit(max_period_count=1)
    assert len(periods) == 1
    assert periods[0] == 175


def test_fourier_find_strongest_period_barthann_window_func():
    data = co2.load().data.resample("ME").mean().ffill()
    period_finder = FourierPeriodFinder(data)
    periods = period_finder.fit(max_period_count=1, window_func="barthann")
    assert len(periods) == 1
    assert periods[0] == 12


def test_fourier_find_strongest_period_bartlett_window_func():
    data = co2.load().data.resample("ME").mean().ffill()
    period_finder = FourierPeriodFinder(data)
    periods = period_finder.fit(max_period_count=1, window_func="bartlett")
    assert len(periods) == 1
    assert periods[0] == 12


def test_fourier_find_strongest_period_blackman_window_func():
    data = co2.load().data.resample("ME").mean().ffill()
    period_finder = FourierPeriodFinder(data)
    periods = period_finder.fit(max_period_count=1, window_func="blackman")
    assert len(periods) == 1
    assert periods[0] == 12


def test_fourier_find_strongest_period_blackmanharris_window_func():
    data = co2.load().data.resample("ME").mean().ffill()
    period_finder = FourierPeriodFinder(data)
    periods = period_finder.fit(max_period_count=1, window_func="blackmanharris")
    assert len(periods) == 1
    assert periods[0] == 12


def test_fourier_find_strongest_period_boxcar_window_func():
    data = co2.load().data.resample("ME").mean().ffill()
    period_finder = FourierPeriodFinder(data)
    periods = period_finder.fit(max_period_count=1, window_func="boxcar")
    assert len(periods) == 1
    assert periods[0] == 175


def test_fourier_find_strongest_period_hamming_window_func():
    data = co2.load().data.resample("ME").mean().ffill()
    period_finder = FourierPeriodFinder(data)
    periods = period_finder.fit(max_period_count=1, window_func="hamming")
    assert len(periods) == 1
    assert periods[0] == 12


def test_fourier_find_strongest_period_hann_window_func():
    data = co2.load().data.resample("ME").mean().ffill()
    period_finder = FourierPeriodFinder(data)
    periods = period_finder.fit(max_period_count=1, window_func="hann")
    assert len(periods) == 1
    assert periods[0] == 12


def test_fourier_find_strongest_period_tukey_window_func():
    data = co2.load().data.resample("ME").mean().ffill()
    period_finder = FourierPeriodFinder(data)
    periods = period_finder.fit(max_period_count=1, window_func="tukey")
    assert len(periods) == 1
    assert periods[0] == 12
