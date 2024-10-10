<div align="center">
<h1>Pyriodicity</h1>

[![PyPI Version](https://img.shields.io/pypi/v/pyriodicity.svg?label=PyPI)](https://pypi.org/project/pyriodicity/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyriodicity?label=Python)
![GitHub License](https://img.shields.io/github/license/iskandergaba/pyriodicity?label=License)
[![CI Build](https://github.com/iskandergaba/pyriodicity/actions/workflows/ci.yml/badge.svg)](https://github.com/iskandergaba/pyriodicity/actions/workflows/ci.yml)
[![Codecov](https://codecov.io/gh/iskandergaba/pyriodicity/graph/badge.svg?token=D5F3PKSOEK)](https://codecov.io/gh/iskandergaba/pyriodicity)
</div>


## About Pyriodicity
Pyriodicity provides intuitive and easy-to-use Python implementation for periodicity (seasonality) detection in univariate time series. Pyriodicity supports the following detection methods:
- [Autocorrelation Function (ACF)](https://otexts.com/fpp3/acf.html)
- [Autoperiod](https://doi.org/10.1137/1.9781611972757.40)
- [CFD-Autoperiod](https://doi.org/10.1007/978-3-030-39098-3_4)
- [Fast Fourier Transform (FFT)](https://otexts.com/fpp3/useful-predictors.html#fourier-series)

## Installation
To install the latest version of `pyriodicity`, simply run:

```shell
pip install pyriodicity
```

## Example
Start by loading a the `co2` emissions sample time series data from [`statsmodels`](https://www.statsmodels.org)
```python
>>> from statsmodels.datasets import co2
>>> data = co2.load().data
```

You can then resample the data to whatever frequency you want. In this example, we downsample the data to a monthly frequency
```python
>>> data = data.resample("ME").mean().ffill()
```

Use `Autoperiod` to find the list of periods based in this data (if any).
```python
>>> from pyriodicity import Autoperiod
>>> autoperiod = Autoperiod(data)
>>> autoperiod.fit()
array([12])
```

The detected periodicity length is 12 which suggests a strong yearly seasonality given that the data has a monthly frequency.

You can also use `CFDAutoperiod` variant of `Autoperiod` or any other supported periodicity detection method such as `ACFPeriodicityDetector` and `FFTPeriodicityDetector` and compare results and performances.

## Development Environment Setup
This project is built and published using [Poetry](https://python-poetry.org). To setup a development environment for this project you can follow these steps:

1. Install one of the compatible [Python](https://www.python.org) versions indicated above.
2. Install [Poetry](https://python-poetry.org/docs/#installing-with-pipx).
3. Navigate to the root folder and install dependencies in a virtual environment:
```shell
poetry install
```
4. If everything worked properly, you should have an environment under the name `pyriodicity-py3.*` activated. You can verify this by running:
```shell
poetry env list
```
5. You can run tests using the command:
```shell
poetry run pytest
```
6. To export the detailed dependency list, consider running the following:
```shell
# Add poetry-plugin-export plugin to poetry
poetry self add poetry-plugin-export

# Export the package dependencies to requirements.txt
poetry export --output requirements.txt

# If you wish to export all the dependencies, including those needed for testing, run the following command
poetry export --with test --output requirements-dev.txt
```

## References
- [1] Hyndman, R.J., & Athanasopoulos, G. (2021) Forecasting: principles and practice, 3rd edition, OTexts: Melbourne, Australia. [OTexts.com/fpp3](https://otexts.com/fpp3). Accessed on 09-15-2024.
- [2] Vlachos, M., Yu, P., & Castelli, V. (2005). On periodicity detection and Structural Periodic similarity. Proceedings of the 2005 SIAM International Conference on Data Mining. [doi.org/10.1137/1.9781611972757.40](https://doi.org/10.1137/1.9781611972757.40).
- [3] Puech, T., Boussard, M., D'Amato, A., & Millerand, G. (2020). A fully automated periodicity detection in time series. In Advanced Analytics and Learning on Temporal Data: 4th ECML PKDD Workshop, AALTD 2019, Würzburg, Germany, September 20, 2019, Revised Selected Papers 4 (pp. 43-54). Springer International Publishing. [doi.org/10.1007/978-3-030-39098-3_4](https://doi.org/10.1007/978-3-030-39098-3_4).
