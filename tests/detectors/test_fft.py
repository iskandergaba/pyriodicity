from pyriodicity import FFTPeriodicityDetector


def test_sinewave_10_fft_find_all_periods(sinewave_10):
    data = sinewave_10
    periods = FFTPeriodicityDetector.detect(data)
    assert len(periods) > 0
    assert 10 in periods


def test_sinewave_50_fft_find_all_periods(sinewave_50):
    data = sinewave_50
    periods = FFTPeriodicityDetector.detect(data)
    assert len(periods) > 0
    assert 50 in periods


def test_sinewave_100_fft_find_all_periods(sinewave_100):
    data = sinewave_100
    periods = FFTPeriodicityDetector.detect(data)
    assert len(periods) > 0
    assert 100 in periods


def test_sinewave_10_fft_find_strongest_period(sinewave_10):
    data = sinewave_10
    periods = FFTPeriodicityDetector.detect(data, max_period_count=1)
    assert len(periods) == 1
    assert 10 in periods


def test_sinewave_50_fft_find_strongest_period(sinewave_50):
    data = sinewave_50
    periods = FFTPeriodicityDetector.detect(data, max_period_count=1)
    assert len(periods) == 1
    assert 50 in periods


def test_sinewave_100_fft_find_strongest_period(sinewave_100):
    data = sinewave_100
    periods = FFTPeriodicityDetector.detect(data, max_period_count=1)
    assert len(periods) == 1
    assert 100 in periods


def test_trianglewave_10_fft_find_all_periods(trianglewave_10):
    data = trianglewave_10
    periods = FFTPeriodicityDetector.detect(data)
    assert len(periods) > 0
    assert 10 in periods


def test_trianglewave_50_fft_find_all_periods(trianglewave_50):
    data = trianglewave_50
    periods = FFTPeriodicityDetector.detect(data)
    assert len(periods) > 0
    assert 50 in periods


def test_trianglewave_100_fft_find_all_periods(trianglewave_100):
    data = trianglewave_100
    periods = FFTPeriodicityDetector.detect(data)
    assert len(periods) > 0
    assert 100 in periods


def test_trianglewave_10_fft_find_strongest_period(trianglewave_10):
    data = trianglewave_10
    periods = FFTPeriodicityDetector.detect(data, max_period_count=1)
    assert len(periods) == 1
    assert 10 in periods


def test_trianglewave_50_fft_find_strongest_period(trianglewave_50):
    data = trianglewave_50
    periods = FFTPeriodicityDetector.detect(data, max_period_count=1)
    assert len(periods) == 1
    assert 50 in periods


def test_trianglewave_100_fft_find_strongest_period(trianglewave_100):
    data = trianglewave_100
    periods = FFTPeriodicityDetector.detect(data, max_period_count=1)
    assert len(periods) == 1
    assert 100 in periods


def test_co2_monthly_fft_find_all_periods(co2_monthly):
    data = co2_monthly
    periods = FFTPeriodicityDetector.detect(data)
    assert len(periods) > 0


def test_co2_monthly_fft_find_first_two_periods(co2_monthly):
    data = co2_monthly
    periods = FFTPeriodicityDetector.detect(data, max_period_count=2)
    assert len(periods) == 2


def test_co2_daily_fft_find_strongest_period(co2_daily):
    data = co2_daily
    periods = FFTPeriodicityDetector.detect(data, max_period_count=1)
    assert len(periods) == 1
    assert 363 in periods


def test_co2_weekly_fft_find_strongest_period(co2_weekly):
    data = co2_weekly
    periods = FFTPeriodicityDetector.detect(data, max_period_count=1)
    assert len(periods) == 1
    assert 52 in periods


def test_co2_monthly_fft_find_strongest_period(co2_monthly):
    data = co2_monthly
    periods = FFTPeriodicityDetector.detect(data, max_period_count=1)
    assert len(periods) == 1
    assert 12 in periods


def test_co2_daily_fft_find_strongest_period_window_func_barthann(co2_daily):
    data = co2_daily
    periods = FFTPeriodicityDetector.detect(
        data, max_period_count=1, window_func="barthann"
    )
    assert len(periods) == 1
    assert 363 in periods


def test_co2_weekly_fft_find_strongest_period_window_func_barthann(co2_weekly):
    data = co2_weekly
    periods = FFTPeriodicityDetector.detect(
        data, max_period_count=1, window_func="barthann"
    )
    assert len(periods) == 1
    assert 52 in periods


def test_co2_monthly_fft_find_strongest_period_window_func_barthann(co2_monthly):
    data = co2_monthly
    periods = FFTPeriodicityDetector.detect(
        data, max_period_count=1, window_func="barthann"
    )
    assert len(periods) == 1
    assert 12 in periods


def test_co2_daily_fft_find_strongest_period_window_func_bartlett(co2_daily):
    data = co2_daily
    periods = FFTPeriodicityDetector.detect(
        data, max_period_count=1, window_func="bartlett"
    )
    assert len(periods) == 1
    assert 363 in periods


def test_co2_weekly_fft_find_strongest_period_window_func_bartlett(co2_weekly):
    data = co2_weekly
    periods = FFTPeriodicityDetector.detect(
        data, max_period_count=1, window_func="bartlett"
    )
    assert len(periods) == 1
    assert 52 in periods


def test_co2_monthly_fft_find_strongest_period_window_func_bartlett(co2_monthly):
    data = co2_monthly
    periods = FFTPeriodicityDetector.detect(
        data, max_period_count=1, window_func="bartlett"
    )
    assert len(periods) == 1
    assert 12 in periods


def test_co2_daily_fft_find_strongest_period_window_func_blackman(co2_daily):
    data = co2_daily
    periods = FFTPeriodicityDetector.detect(
        data, max_period_count=1, window_func="blackman"
    )
    assert len(periods) == 1
    assert 363 in periods


def test_co2_weekly_fft_find_strongest_period_window_func_blackman(co2_weekly):
    data = co2_weekly
    periods = FFTPeriodicityDetector.detect(
        data, max_period_count=1, window_func="blackman"
    )
    assert len(periods) == 1
    assert 52 in periods


def test_co2_monthly_fft_find_strongest_period_window_func_blackman(co2_monthly):
    data = co2_monthly
    periods = FFTPeriodicityDetector.detect(
        data, max_period_count=1, window_func="blackman"
    )
    assert len(periods) == 1
    assert 12 in periods


def test_co2_daily_fft_find_strongest_period_window_func_blackmanharris(co2_daily):
    data = co2_daily
    periods = FFTPeriodicityDetector.detect(
        data, max_period_count=1, window_func="blackmanharris"
    )
    assert len(periods) == 1
    assert 363 in periods


def test_co2_weekly_fft_find_strongest_period_window_func_blackmanharris(co2_weekly):
    data = co2_weekly
    periods = FFTPeriodicityDetector.detect(
        data, max_period_count=1, window_func="blackmanharris"
    )
    assert len(periods) == 1
    assert 52 in periods


def test_co2_monthly_fft_find_strongest_period_window_func_blackmanharris(co2_monthly):
    data = co2_monthly
    periods = FFTPeriodicityDetector.detect(
        data, max_period_count=1, window_func="blackmanharris"
    )
    assert len(periods) == 1
    assert 12 in periods


def test_co2_daily_fft_find_strongest_period_window_func_boxcar(co2_daily):
    data = co2_daily
    periods = FFTPeriodicityDetector.detect(
        data, max_period_count=1, window_func="boxcar"
    )
    assert len(periods) == 1
    assert 363 in periods


def test_co2_weekly_fft_find_strongest_period_window_func_boxcar(co2_weekly):
    data = co2_weekly
    periods = FFTPeriodicityDetector.detect(
        data, max_period_count=1, window_func="boxcar"
    )
    assert len(periods) == 1
    assert 52 in periods


def test_co2_monthly_fft_find_strongest_period_window_func_boxcar(co2_monthly):
    data = co2_monthly
    periods = FFTPeriodicityDetector.detect(
        data, max_period_count=1, window_func="boxcar"
    )
    assert len(periods) == 1
    assert 12 in periods


def test_co2_daily_fft_find_strongest_period_window_func_hamming(co2_daily):
    data = co2_daily
    periods = FFTPeriodicityDetector.detect(
        data, max_period_count=1, window_func="hamming"
    )
    assert len(periods) == 1
    assert 363 in periods


def test_co2_weekly_fft_find_strongest_period_window_func_hamming(co2_weekly):
    data = co2_weekly
    periods = FFTPeriodicityDetector.detect(
        data, max_period_count=1, window_func="hamming"
    )
    assert len(periods) == 1
    assert 52 in periods


def test_co2_monthly_fft_find_strongest_period_window_func_hamming(co2_monthly):
    data = co2_monthly
    periods = FFTPeriodicityDetector.detect(
        data, max_period_count=1, window_func="hamming"
    )
    assert len(periods) == 1
    assert 12 in periods


def test_co2_daily_fft_find_strongest_period_window_func_hann(co2_daily):
    data = co2_daily
    periods = FFTPeriodicityDetector.detect(
        data, max_period_count=1, window_func="hann"
    )
    assert len(periods) == 1
    assert 363 in periods


def test_co2_weekly_fft_find_strongest_period_window_func_hann(co2_weekly):
    data = co2_weekly
    periods = FFTPeriodicityDetector.detect(
        data, max_period_count=1, window_func="hann"
    )
    assert len(periods) == 1
    assert 52 in periods


def test_co2_monthly_fft_find_strongest_period_window_func_hann(co2_monthly):
    data = co2_monthly
    periods = FFTPeriodicityDetector.detect(
        data, max_period_count=1, window_func="hann"
    )
    assert len(periods) == 1
    assert 12 in periods


def test_co2_daily_fft_find_strongest_period_window_func_tukey(co2_daily):
    data = co2_daily
    periods = FFTPeriodicityDetector.detect(
        data, max_period_count=1, window_func="tukey"
    )
    assert len(periods) == 1
    assert 363 in periods


def test_co2_weekly_fft_find_strongest_period_window_func_tukey(co2_weekly):
    data = co2_weekly
    periods = FFTPeriodicityDetector.detect(
        data, max_period_count=1, window_func="tukey"
    )
    assert len(periods) == 1
    assert 52 in periods


def test_co2_monthly_fft_find_strongest_period_window_func_tukey(co2_monthly):
    data = co2_monthly
    periods = FFTPeriodicityDetector.detect(
        data, max_period_count=1, window_func="tukey"
    )
    assert len(periods) == 1
    assert 12 in periods
