from pyriodicity import OnlineFFTPeriodicityDetector


def test_sinewave_10_online_fft_find_strongest_period_window_size_200(
    sinewave_10_generator,
):
    detector = OnlineFFTPeriodicityDetector(window_size=200, max_period_count=1)
    for sample in sinewave_10_generator:
        periods = detector.detect(sample)
    assert len(periods) > 0
    assert 10 in periods


def test_sinewave_50_online_fft_find_strongest_period_window_size_200(
    sinewave_50_generator,
):
    detector = OnlineFFTPeriodicityDetector(window_size=200, max_period_count=1)
    for sample in sinewave_50_generator:
        periods = detector.detect(sample)
    assert len(periods) > 0
    assert 50 in periods


def test_sinewave_100_online_fft_find_strongest_period_window_size_200(
    sinewave_100_generator,
):
    detector = OnlineFFTPeriodicityDetector(window_size=200, max_period_count=1)
    for sample in sinewave_100_generator:
        periods = detector.detect(sample)
    assert len(periods) > 0
    assert 100 in periods


def test_trianglewave_10_online_fft_find_strongest_period_window_size_200(
    trianglewave_10_generator,
):
    detector = OnlineFFTPeriodicityDetector(window_size=200, max_period_count=1)
    for sample in trianglewave_10_generator:
        periods = detector.detect(sample)
    assert len(periods) > 0
    assert 10 in periods


def test_trianglewave_50_online_fft_find_strongest_period_window_size_200(
    trianglewave_50_generator,
):
    detector = OnlineFFTPeriodicityDetector(window_size=200, max_period_count=1)
    for sample in trianglewave_50_generator:
        periods = detector.detect(sample)
    assert len(periods) > 0
    assert 50 in periods


def test_trianglewave_100_online_fft_find_strongest_period_window_size_200(
    trianglewave_100_generator,
):
    detector = OnlineFFTPeriodicityDetector(window_size=200, max_period_count=1)
    for sample in trianglewave_100_generator:
        periods = detector.detect(sample)
    assert len(periods) > 0
    assert 100 in periods


def test_co2_monthly_online_fft_find_first_two_periods_window_size_100(
    co2_monthly_generator,
):
    detector = OnlineFFTPeriodicityDetector(window_size=100, max_period_count=2)
    for sample in co2_monthly_generator:
        periods = detector.detect(sample)
    assert len(periods) == 2


def test_co2_weekly_online_fft_find_all_periods_window_size_1024(
    co2_weekly_generator,
):
    detector = OnlineFFTPeriodicityDetector(window_size=1024)
    for sample in co2_weekly_generator:
        periods = detector.detect(sample)
    assert len(periods) > 0
    assert 51 in periods


def test_co2_monthly_online_fft_find_all_periods_window_size_512(
    co2_monthly_generator,
):
    detector = OnlineFFTPeriodicityDetector(window_size=512)
    for sample in co2_monthly_generator:
        periods = detector.detect(sample)
    assert len(periods) > 0
    assert 12 in periods


def test_co2_weekly_online_fft_find_all_periods_window_size_1024_window_func_blackman(
    co2_weekly_generator,
):
    detector = OnlineFFTPeriodicityDetector(window_size=1024, window_func="blackman")
    for sample in co2_weekly_generator:
        periods = detector.detect(sample)
    assert len(periods) > 0
    assert 51 in periods


def test_co2_monthly_online_fft_find_all_periods_window_size_512_window_func_blackman(
    co2_monthly_generator,
):
    detector = OnlineFFTPeriodicityDetector(window_size=512, window_func="blackman")
    for sample in co2_monthly_generator:
        periods = detector.detect(sample)
    assert len(periods) > 0
    assert 12 in periods
