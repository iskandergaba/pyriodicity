import pytest

from pyriodicity import OnlineACFPeriodicityDetector


def test_sinewave_10_online_acf_find_all_periods_window_size_100_buffer_size_50():
    with pytest.raises(ValueError):
        OnlineACFPeriodicityDetector(window_size=100, buffer_size=50)


def test_sinewave_10_online_acf_find_first_two_periods_window_size_50_buffer_size_150(
    sinewave_10_generator,
):
    detector = OnlineACFPeriodicityDetector(window_size=50, buffer_size=150)
    for sample in sinewave_10_generator:
        periods = detector.detect(sample, max_period_count=2)
    assert len(periods) <= 2
    assert 10 in periods


def test_sinewave_50_online_acf_find_strongest_period_window_size_150_buffer_size_400(
    sinewave_50_generator,
):
    detector = OnlineACFPeriodicityDetector(window_size=150, buffer_size=400)
    for sample in sinewave_50_generator:
        periods = detector.detect(sample, max_period_count=1)
    assert len(periods) > 0
    assert 50 in periods


def test_sinewave_100_online_acf_find_strongest_period_window_size_300(
    sinewave_100_generator,
):
    detector = OnlineACFPeriodicityDetector(window_size=300)
    for sample in sinewave_100_generator:
        periods = detector.detect(sample, max_period_count=None)
    assert len(periods) > 0
    assert 99 in periods


def test_trianglewave_10_online_acf_find_all_periods_window_size_50_buffer_size_150(
    trianglewave_10_generator,
):
    detector = OnlineACFPeriodicityDetector(window_size=50, buffer_size=150)
    for sample in trianglewave_10_generator:
        periods = detector.detect(sample)
    assert len(periods) > 0
    assert 11 in periods


def test_trianglewave_50_online_acf_find_all_periods_window_size_150_buffer_size_400(
    trianglewave_50_generator,
):
    detector = OnlineACFPeriodicityDetector(window_size=150, buffer_size=400)
    for sample in trianglewave_50_generator:
        periods = detector.detect(sample)
    assert len(periods) > 0
    assert 50 in periods


def test_trianglewave_100_online_acf_find_all_periods_window_size_300(
    trianglewave_100_generator,
):
    detector = OnlineACFPeriodicityDetector(window_size=300)
    for sample in trianglewave_100_generator:
        periods = detector.detect(sample)
    assert len(periods) > 0
    assert 100 in periods


def test_co2_weekly_online_acf_find_all_periods_window_size_256(
    co2_weekly_generator,
):
    detector = OnlineACFPeriodicityDetector(window_size=256)
    for sample in co2_weekly_generator:
        periods = detector.detect(sample)
    assert len(periods) > 0
    assert 52 in periods


def test_co2_monthly_online_acf_find_all_periods_window_size_128(
    co2_monthly_generator,
):
    detector = OnlineACFPeriodicityDetector(window_size=128)
    for sample in co2_monthly_generator:
        periods = detector.detect(sample)
    assert len(periods) > 0
    assert 12 in periods


def test_co2_weekly_online_acf_find_all_periods_window_size_128_window_func_blackman(
    co2_weekly_generator,
):
    detector = OnlineACFPeriodicityDetector(window_size=128, window_func="blackman")
    for sample in co2_weekly_generator:
        periods = detector.detect(sample)
    assert len(periods) > 0
    assert 52 in periods


def test_co2_monthly_online_acf_find_all_periods_window_size_64_window_func_blackman(
    co2_monthly_generator,
):
    detector = OnlineACFPeriodicityDetector(window_size=64, window_func="blackman")
    for sample in co2_monthly_generator:
        periods = detector.detect(sample)
    assert len(periods) > 0
    assert 12 in periods
