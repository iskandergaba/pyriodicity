import pytest
from statsmodels.datasets import co2


@pytest.fixture(scope="module")
def co2_daily():
    return co2.load().data.resample("D").mean().ffill()


@pytest.fixture(scope="module")
def co2_weekly():
    return co2.load().data.resample("W").mean().ffill()


@pytest.fixture(scope="module")
def co2_monthly():
    return co2.load().data.resample("ME").mean().ffill()
