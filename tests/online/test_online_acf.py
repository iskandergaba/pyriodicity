import pytest

from pyriodicity import OnlineACFPeriodicityDetector


def test_online_acf_find_all_periods_window_size_100_buffer_size_50():
    with pytest.raises(ValueError):
        OnlineACFPeriodicityDetector(window_size=100, buffer_size=50)


def test_trianglewave_10_online_fft_find_two_periods_window_size_50_buffer_size_150(
    trianglewave_10_stream,
):
    detector = OnlineACFPeriodicityDetector(window_size=50, buffer_size=150)
    for sample in trianglewave_10_stream:
        periods = detector.detect(sample, max_period_count=2)
    assert len(periods) <= 2
    assert any(period == pytest.approx(10, abs=1) for period in periods)


def test_trianglewave_50_online_acf_find_all_periods_window_size_150_buffer_size_400(
    trianglewave_50_stream,
):
    detector = OnlineACFPeriodicityDetector(window_size=150, buffer_size=400)
    for sample in trianglewave_50_stream:
        periods = detector.detect(sample)
    assert len(periods) > 0
    assert any(period == pytest.approx(50, abs=1) for period in periods)


def test_trianglewave_100_online_acf_find_strongest_period_window_size_300(
    trianglewave_100_stream,
):
    detector = OnlineACFPeriodicityDetector(window_size=300)
    for sample in trianglewave_100_stream:
        periods = detector.detect(sample, max_period_count=1)
    assert len(periods) > 0
    assert any(period == pytest.approx(100, abs=1) for period in periods)


def test_sinewave_10_batch_size_100_online_fft_find_all_periods_window_size_50(
    sinewave_10_batch_size_100_generator,
):
    detector = OnlineACFPeriodicityDetector(window_size=50)
    for sample in sinewave_10_batch_size_100_generator:
        periods = detector.detect(sample)
    assert len(periods) > 0
    assert any(period == pytest.approx(10, abs=1) for period in periods)


def test_sinewave_50_batch_size_100_online_fft_find_all_periods_window_size_200(
    sinewave_50_batch_size_100_generator,
):
    detector = OnlineACFPeriodicityDetector(window_size=200)
    for sample in sinewave_50_batch_size_100_generator:
        periods = detector.detect(sample)
    assert len(periods) > 0
    assert any(period == pytest.approx(50, abs=1) for period in periods)


def test_sinewave_100_batch_size_100_online_fft_find_all_periods_window_size_300(
    sinewave_100_batch_size_100_generator,
):
    detector = OnlineACFPeriodicityDetector(window_size=300)
    for sample in sinewave_100_batch_size_100_generator:
        periods = detector.detect(sample)
    assert len(periods) > 0
    assert any(period == pytest.approx(100, abs=1) for period in periods)


def test_co2_weekly_online_acf_find_all_periods_window_size_256(
    co2_weekly_stream,
):
    detector = OnlineACFPeriodicityDetector(window_size=256)
    for sample in co2_weekly_stream:
        periods = detector.detect(sample)
    assert len(periods) > 0
    assert any(period == pytest.approx(52, abs=1) for period in periods)


def test_co2_monthly_online_acf_find_all_periods_window_size_128(
    co2_monthly_stream,
):
    detector = OnlineACFPeriodicityDetector(window_size=128)
    for sample in co2_monthly_stream:
        periods = detector.detect(sample)
    assert len(periods) > 0
    assert any(period == pytest.approx(12, abs=1) for period in periods)


def test_co2_weekly_batch_size_100_online_acf_find_all_periods_window_size_256(
    co2_weekly_batch_size_100_generator,
):
    detector = OnlineACFPeriodicityDetector(window_size=256)
    for sample in co2_weekly_batch_size_100_generator:
        periods = detector.detect(sample)
    assert len(periods) > 0
    assert any(period == pytest.approx(52, abs=1) for period in periods)


def test_co2_monthly_batch_size_10_online_acf_find_all_periods_window_size_128(
    co2_monthly_batch_size_10_generator,
):
    detector = OnlineACFPeriodicityDetector(window_size=128)
    for sample in co2_monthly_batch_size_10_generator:
        periods = detector.detect(sample)
    assert len(periods) > 0
    assert any(period == pytest.approx(12, abs=1) for period in periods)


def test_co2_weekly_online_acf_find_all_periods_window_size_256_window_func_blackman(
    co2_weekly_stream,
):
    detector = OnlineACFPeriodicityDetector(window_size=256, window_func="blackman")
    for sample in co2_weekly_stream:
        periods = detector.detect(sample)
    assert len(periods) > 0
    assert any(period == pytest.approx(52, abs=1) for period in periods)


def test_co2_monthly_online_acf_find_all_periods_window_size_128_window_func_blackman(
    co2_monthly_stream,
):
    detector = OnlineACFPeriodicityDetector(window_size=128, window_func="blackman")
    for sample in co2_monthly_stream:
        periods = detector.detect(sample)
    assert len(periods) > 0
    assert any(period == pytest.approx(12, abs=1) for period in periods)
