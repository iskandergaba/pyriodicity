import numpy as np
import pytest
from scipy.signal import sawtooth
from statsmodels.datasets import co2


@pytest.fixture(scope="module")
def sinewave_10():
    return sinewave(1000, 10, 1)


@pytest.fixture(scope="module")
def sinewave_50():
    return sinewave(1000, 50, 1)


@pytest.fixture(scope="module")
def sinewave_100():
    return sinewave(1000, 100, 1)


@pytest.fixture
def sinewave_10_stream():
    return (sample for sample in sinewave(1000, 10, 1))


@pytest.fixture
def sinewave_50_stream():
    return (sample for sample in sinewave(1000, 50, 1))


@pytest.fixture
def sinewave_100_stream():
    return (sample for sample in sinewave(1000, 100, 1))


@pytest.fixture
def sinewave_10_batch_size_100_generator():
    return data_generator(sinewave(1000, 10, 1), batch_size=100)


@pytest.fixture
def sinewave_50_batch_size_100_generator():
    return data_generator(sinewave(1000, 50, 1), batch_size=100)


@pytest.fixture
def sinewave_100_batch_size_100_generator():
    return data_generator(sinewave(1000, 100, 1), batch_size=100)


@pytest.fixture(scope="module")
def trianglewave_10():
    return trianglewave(1000, 10, 1)


@pytest.fixture(scope="module")
def trianglewave_50():
    return trianglewave(1000, 50, 1)


@pytest.fixture(scope="module")
def trianglewave_100():
    return trianglewave(1000, 100, 1)


@pytest.fixture
def trianglewave_10_stream():
    return (sample for sample in trianglewave(1000, 10, 1))


@pytest.fixture
def trianglewave_50_stream():
    return (sample for sample in trianglewave(1000, 50, 1))


@pytest.fixture
def trianglewave_100_stream():
    return (sample for sample in trianglewave(1000, 100, 1))


@pytest.fixture
def trianglewave_10_batch_size_100_generator():
    return data_generator(trianglewave(1000, 10, 1), batch_size=100)


@pytest.fixture
def trianglewave_50_batch_size_100_generator():
    return data_generator(trianglewave(1000, 50, 1), batch_size=100)


@pytest.fixture
def trianglewave_100_batch_size_100_generator():
    return data_generator(trianglewave(1000, 100, 1), batch_size=100)


@pytest.fixture(scope="module")
def co2_weekly():
    return co2_data()


@pytest.fixture(scope="module")
def co2_monthly():
    return co2_data().resample("ME").mean()


@pytest.fixture
def co2_weekly_stream():
    return (sample for sample in co2_data().values)


@pytest.fixture
def co2_monthly_stream():
    return (sample for sample in co2_data().resample("ME").mean().values)


@pytest.fixture
def co2_weekly_batch_size_100_generator():
    return data_generator(co2_data().values, batch_size=100)


@pytest.fixture
def co2_monthly_batch_size_10_generator():
    return data_generator(co2_data().resample("ME").mean().values, batch_size=10)


def sinewave(n, period, amp):
    x = np.arange(0, n, 1)
    freq = 1 / period
    return amp * np.sin(2 * np.pi * freq * x)


def trianglewave(n, period, amp):
    x = np.arange(0, n, 1)
    freq = 1 / period
    return amp * sawtooth(2 * np.pi * freq * x)


def co2_data():
    return co2.load().data.ffill()


def data_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield from data[i : i + batch_size]
