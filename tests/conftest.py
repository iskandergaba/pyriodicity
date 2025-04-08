from typing import Generator

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
def sinewave_10_generator() -> Generator[float, None, None]:
    return (sample for sample in sinewave(10000, 10, 1))


@pytest.fixture
def sinewave_50_generator() -> Generator[float, None, None]:
    return (sample for sample in sinewave(10000, 50, 1))


@pytest.fixture
def sinewave_100_generator() -> Generator[float, None, None]:
    return (sample for sample in sinewave(10000, 100, 1))


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
def trianglewave_10_generator() -> Generator[float, None, None]:
    return (sample for sample in trianglewave(10000, 10, 1))


@pytest.fixture
def trianglewave_50_generator() -> Generator[float, None, None]:
    return (sample for sample in trianglewave(10000, 50, 1))


@pytest.fixture
def trianglewave_100_generator() -> Generator[float, None, None]:
    return (sample for sample in trianglewave(10000, 100, 1))


@pytest.fixture(scope="module")
def co2_weekly():
    return co2_data()


@pytest.fixture(scope="module")
def co2_monthly():
    return co2_data().resample("ME").mean()


@pytest.fixture
def co2_weekly_generator():
    return (sample for sample in co2_data().values)


@pytest.fixture
def co2_monthly_generator():
    return (sample for sample in co2_data().resample("ME").mean().values)


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
