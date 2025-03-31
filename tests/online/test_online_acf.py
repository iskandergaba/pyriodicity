from pyriodicity import OnlineACFPeriodicityDetector


def test_sinewave_10_online_acf_find_strongest_period_window_size_100(
    sinewave_10_generator,
):
    detector = OnlineACFPeriodicityDetector(window_size=100, max_period_count=1)
    for sample in sinewave_10_generator:
        periods = detector.detect(sample)
    assert len(periods) > 0
    assert 10 in periods


def test_sinewave_50_online_acf_find_strongest_period_window_size_200(
    sinewave_50_generator,
):
    detector = OnlineACFPeriodicityDetector(window_size=200, max_period_count=1)
    for sample in sinewave_50_generator:
        periods = detector.detect(sample)
    assert len(periods) > 0
    assert 50 in periods


def test_sinewave_100_online_acf_find_strongest_period_window_size_300(
    sinewave_100_generator,
):
    detector = OnlineACFPeriodicityDetector(window_size=300, max_period_count=1)
    for sample in sinewave_100_generator:
        periods = detector.detect(sample)
    assert len(periods) > 0
    assert 99 in periods


def test_trianglewave_10_online_acf_find_all_periods_window_size_100(
    trianglewave_10_generator,
):
    detector = OnlineACFPeriodicityDetector(window_size=100)
    for sample in trianglewave_10_generator:
        periods = detector.detect(sample)
    assert len(periods) > 0
    assert 10 in periods


def test_trianglewave_50_online_acf_find_all_periods_window_size_200(
    trianglewave_50_generator,
):
    detector = OnlineACFPeriodicityDetector(window_size=200)
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
