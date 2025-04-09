import pytest

from pyriodicity import OnlineFFTPeriodicityDetector


def test_sinewave_10_online_fft_find_all_periods_window_size_100_buffer_size_50():
    with pytest.raises(ValueError):
        OnlineFFTPeriodicityDetector(window_size=100, buffer_size=50)


def test_sinewave_10_online_fft_find_strongest_period_window_size_50(
    sinewave_10_generator,
):
    detector = OnlineFFTPeriodicityDetector(window_size=50)
    for sample in sinewave_10_generator:
        periods = detector.detect(sample, max_period_count=1)
    assert len(periods) > 0
    assert 10 in periods


def test_sinewave_50_online_fft_find_strongest_period_window_size_200(
    sinewave_50_generator,
):
    detector = OnlineFFTPeriodicityDetector(window_size=200)
    for sample in sinewave_50_generator:
        periods = detector.detect(sample, max_period_count=1)
    assert len(periods) > 0
    assert 50 in periods


def test_sinewave_100_online_fft_find_strongest_period_window_size_300(
    sinewave_100_generator,
):
    detector = OnlineFFTPeriodicityDetector(window_size=300)
    for sample in sinewave_100_generator:
        periods = detector.detect(sample, max_period_count=1)
    assert len(periods) > 0
    assert 100 in periods


def test_trianglewave_10_online_fft_find_strongest_period_window_size_50(
    trianglewave_10_generator,
):
    detector = OnlineFFTPeriodicityDetector(window_size=50)
    for sample in trianglewave_10_generator:
        periods = detector.detect(sample, max_period_count=None)
    assert len(periods) > 0
    assert 10 in periods


def test_trianglewave_50_online_fft_find_strongest_period_window_size_200(
    trianglewave_50_generator,
):
    detector = OnlineFFTPeriodicityDetector(window_size=200)
    for sample in trianglewave_50_generator:
        periods = detector.detect(sample, max_period_count=1)
    assert len(periods) > 0
    assert 50 in periods


def test_trianglewave_100_online_fft_find_strongest_period_window_size_300(
    trianglewave_100_generator,
):
    detector = OnlineFFTPeriodicityDetector(window_size=300)
    for sample in trianglewave_100_generator:
        periods = detector.detect(sample, max_period_count=1)
    assert len(periods) > 0
    assert 100 in periods


def test_co2_monthly_online_fft_find_first_two_periods_window_size_128(
    co2_monthly_generator,
):
    detector = OnlineFFTPeriodicityDetector(window_size=128)
    for sample in co2_monthly_generator:
        periods = detector.detect(sample, max_period_count=2)
    assert len(periods) == 2


def test_co2_weekly_online_fft_find_all_periods_window_size_256(
    co2_weekly_generator,
):
    detector = OnlineFFTPeriodicityDetector(window_size=256)
    for sample in co2_weekly_generator:
        periods = detector.detect(sample)
    assert len(periods) > 0
    assert 51 in periods


def test_co2_monthly_online_fft_find_all_periods_window_size_128(
    co2_monthly_generator,
):
    detector = OnlineFFTPeriodicityDetector(window_size=128)
    for sample in co2_monthly_generator:
        periods = detector.detect(sample)
    assert len(periods) > 0
    assert 12 in periods


def test_co2_weekly_online_fft_find_all_periods_window_size_256_window_func_blackman(
    co2_weekly_generator,
):
    detector = OnlineFFTPeriodicityDetector(window_size=256, window_func="blackman")
    for sample in co2_weekly_generator:
        periods = detector.detect(sample)
    assert len(periods) > 0
    assert 51 in periods


def test_co2_monthly_online_fft_find_all_periods_window_size_128_window_func_blackman(
    co2_monthly_generator,
):
    detector = OnlineFFTPeriodicityDetector(window_size=128, window_func="blackman")
    for sample in co2_monthly_generator:
        periods = detector.detect(sample)
    assert len(periods) > 0
    assert 12 in periods
