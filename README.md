<div align="center">
<h1>Pyriodicity</h1>

[![PyPI Version](https://img.shields.io/pypi/v/pyriodicity.svg?label=PyPI)](https://pypi.org/project/pyriodicity/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyriodicity?label=Python)
![License](https://img.shields.io/pypi/l/pyriodicity?label=License)
[![Codecov](https://codecov.io/gh/iskandergaba/pyriodicity/graph/badge.svg?token=D5F3PKSOEK)](https://codecov.io/gh/iskandergaba/pyriodicity)
[![Docs](https://readthedocs.org/projects/pyriodicity/badge/?version=latest)](https://pyriodicity.readthedocs.io/en/latest)
[![CI Build](https://github.com/iskandergaba/pyriodicity/actions/workflows/ci.yml/badge.svg)](https://github.com/iskandergaba/pyriodicity/actions/workflows/ci.yml)

Pyriodicity provides an intuitive and efficient Python implementation of periodicity length detection methods in univariate signals. You can check the supported detection methods in the [API Reference](https://pyriodicity.readthedocs.io/en/stable/api.html).
</div>

## Installation
To install ``pyriodicity``, simply run:
```shell
pip install pyriodicity
```

To install the latest development version, you can run:
```shell
pip install git+https://github.com/iskandergaba/pyriodicity.git
```

## Usage
Please refer to the [package documentation](https://pyriodicity.readthedocs.io) for more information.

For this example, start by loading Mauna Loa Weekly Atmospheric CO2 Data from [`statsmodels`](https://www.statsmodels.org) and downsampling its data to a monthly frequency.
```python
>>> from statsmodels.datasets import co2
>>> data = co2.load().data
>>> data = data.resample("ME").mean().ffill()
```

Use `Autoperiod` to find the list of periodicity lengths in this data, if any.
```python
>>> from pyriodicity import Autoperiod
>>> Autoperiod.detect(data)
array([12])
```

The detected periodicity length is 12 which suggests a strong yearly seasonality given that the data has a monthly frequency.

We can also use online detection methods for data streams as follows.
```python
>>> from pyriodicity import OnlineACFPeriodicityDetector
>>> data_stream = (sample for sample in data.values)
>>> detector = OnlineACFPeriodicityDetector(window_size=128)
>>> for sample in data_stream:
...   periods = detector.detect(sample)
>>> 12 in periods
True
```

All the supported periodicity detection methods can be used in the same manner as in the examples above with different optional parameters. Check the [API Reference](https://pyriodicity.readthedocs.io/en/stable/api.html) for more details.

## References
1. Hyndman, R.J., & Athanasopoulos, G. (2021). Forecasting: principles and practice, 3rd edition, OTexts: Melbourne, Australia. [OTexts.com/fpp3](https://otexts.com/fpp3). Accessed on 09-15-2024.
2. Vlachos, M., Yu, P., & Castelli, V. (2005). On periodicity detection and Structural Periodic similarity. Proceedings of the 2005 SIAM International Conference on Data Mining. [doi.org/10.1137/1.9781611972757.40](https://doi.org/10.1137/1.9781611972757.40).
3. Puech, T., Boussard, M., D'Amato, A., & Millerand, G. (2020). A fully automated periodicity detection in time series. In Advanced Analytics and Learning on Temporal Data: 4th ECML PKDD Workshop, AALTD 2019, Würzburg, Germany, September 20, 2019, Revised Selected Papers 4 (pp. 43-54). Springer International Publishing. [doi.org/10.1007/978-3-030-39098-3_4](https://doi.org/10.1007/978-3-030-39098-3_4).
4. Toller, M., Santos, T., & Kern, R. (2019). SAZED: parameter-free domain-agnostic season length estimation in time series data. Data Mining and Knowledge Discovery, 33(6), 1775-1798. [doi.org/10.1007/s10618-019-00645-z](https://doi.org/10.1007/s10618-019-00645-z).
5. Wen, Q., He, K., Sun, L., Zhang, Y., Ke, M., & Xu, H. (2021, June). RobustPeriod: Robust time-frequency mining for multiple periodicity detection. In Proceedings of the 2021 international conference on management of data (pp. 2328-2337). [doi.org/10.1145/3448016.3452779](https://doi.org/10.1145/3448016.3452779).
