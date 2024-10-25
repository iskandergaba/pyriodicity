<div align="center">
<h1>Pyriodicity</h1>

[![PyPI Version](https://img.shields.io/pypi/v/pyriodicity.svg?label=PyPI)](https://pypi.org/project/pyriodicity/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyriodicity?label=Python)
![GitHub License](https://img.shields.io/github/license/iskandergaba/pyriodicity?label=License)
[![Codecov](https://codecov.io/gh/iskandergaba/pyriodicity/graph/badge.svg?token=D5F3PKSOEK)](https://codecov.io/gh/iskandergaba/pyriodicity)
[![Docs](https://readthedocs.org/projects/pyriodicity/badge/?version=latest)](https://pyriodicity.readthedocs.io/en/latest)
[![CI Build](https://github.com/iskandergaba/pyriodicity/actions/workflows/ci.yml/badge.svg)](https://github.com/iskandergaba/pyriodicity/actions/workflows/ci.yml)
</div>

## About Pyriodicity
Pyriodicity provides an intuitive and easy-to-use Python implementation for periodicity detection in univariate signals. Pyriodicity supports the following detection methods:
- [Autocorrelation Function (ACF)](https://otexts.com/fpp3/acf.html)
- [Autoperiod](https://doi.org/10.1137/1.9781611972757.40)
- [CFD-Autoperiod](https://doi.org/10.1007/978-3-030-39098-3_4)
- [Fast Fourier Transform (FFT)](https://otexts.com/fpp3/useful-predictors.html#fourier-series)

## Installation
To install the latest version of `pyriodicity`, simply run:

```shell
pip install pyriodicity
```

## Usage
Please refer to the [package documentation](https://pyriodicity.readthedocs.io) for more information.

For this example, start by loading Mauna Loa Weekly Atmospheric CO2 Data from [`statsmodels`](https://www.statsmodels.org) and downsampling its data to a monthly frequency.
```python
>>> from statsmodels.datasets import co2
>>> data = co2.load().data
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

All the supported estimation algorithms can be used in the same manner as in the example above with different optional parameters. Check the [API Reference](https://pyriodicity.readthedocs.io/en/stable/api.html) for more details.

## References
- [1] Hyndman, R.J., & Athanasopoulos, G. (2021) Forecasting: principles and practice, 3rd edition, OTexts: Melbourne, Australia. [OTexts.com/fpp3](https://otexts.com/fpp3). Accessed on 09-15-2024.
- [2] Vlachos, M., Yu, P., & Castelli, V. (2005). On periodicity detection and Structural Periodic similarity. Proceedings of the 2005 SIAM International Conference on Data Mining. [doi.org/10.1137/1.9781611972757.40](https://doi.org/10.1137/1.9781611972757.40).
- [3] Puech, T., Boussard, M., D'Amato, A., & Millerand, G. (2020). A fully automated periodicity detection in time series. In Advanced Analytics and Learning on Temporal Data: 4th ECML PKDD Workshop, AALTD 2019, Würzburg, Germany, September 20, 2019, Revised Selected Papers 4 (pp. 43-54). Springer International Publishing. [doi.org/10.1007/978-3-030-39098-3_4](https://doi.org/10.1007/978-3-030-39098-3_4).
