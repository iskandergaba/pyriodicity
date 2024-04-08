# `auto-period-finder`
[![PyPI Version](https://img.shields.io/pypi/v/auto-period-finder.svg?label=PyPI)](https://pypi.org/project/auto-period-finder/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/auto-period-finder?label=Python)
![GitHub License](https://img.shields.io/github/license/iskandergaba/auto-period-finder?label=License)

## About `auto-period-finder`
`auto-period-finder` is an autocorrelation function (ACF) based seasonality periods automatic finder for univariate time series.

## Installation
To install the latest version of `auto-period-finder`, simply run:

```shell
pip install auto-period-finder
```

## Example
Start by loading a timeseries dataset with a frequency. We can use `co2` emissions sample dataset from `statsmodels`
```python
from statsmodels.datasets import co2
data = co2.load().data
```

You can resample the data to whatever frequency you want.

```python
data = data.resample("ME").mean().ffill()
```

Use `AutoPeriodFinder` to find the list of seasonality periods based on ACF.
```python
from auto_period_finder import AutoPeriodFinder
period_finder = AutoPeriodFinder(data)
periods = period_finder.fit()
```

You can also find the most prominent period either ACF-wise:
```python
strongest_period_acf = period_finder.fit_find_strongest_acf()
```

or variance-wise:
```python
strongest_period_var = period_finder.fit_find_strongest_var()
```
You can learn more about calculating seasonality component through variance from [here](OTexts.com/fpp3/stlfeatures.html).


## How to Get Started
This project is built and published using [Poetry](https://python-poetry.org). To setup development environment for this project you can follow these steps:

1. First, you need to install [Python](https://www.python.org) of one of the compatible versions indicated above.
2. Install Poetry. You can follow this [guide](https://python-poetry.org/docs/#installing-with-the-official-installer) and use their official installer.
3. Navigate to the root folder and install dependencies in a virtual environment:
```shell
poetry install
```
4. If everything worked properly, you should have `auto-period-finder-geinoPPi-py3.10` environment activated. You can verify this by running:
```shell
poetry env list
```
5. You can run tests using the command:
```shell
poetry run pytest
```
6. To export the list detailed list of dependencies, run the following command:
```shell
poetry self add poetry-plugin-export
poetry export --output requirements.txt
```

## ACF-Based Seasonality Period Detection Explained
An easy and quick way to find seasonality periods of a univariate time series is to check its autocorrelation function (ACF) and look for specific charecteristics in lag values that we will detail in a second. You can read more information about time series ACF [here](https://otexts.com/fpp3/acf.html), but intuitively, An autocorrelation coefficient $r_k$ measures the the linear relationship between $k$-lagged values of a given time series. In simpler terms, $r_k$ measures how similar/dissimilar time series values that $k$-length apart from each other. The set of $r_k$ values for each lag $k$ makes ACF. Equipped with this information, I developed a package for finding time series seasonality periods automatically using ACF information.

Simply put, given a univariate time series $T$, the algorithm finds, iteratively, lag values $k$ such that:
1. $1 \lt k \leq \frac{\lvert T \rvert}{2}$
2. Autocorrelation coefficients $r_q$ are local maxima where $q \in \{k, 2k, 3k, ...\}$
3. $\forall p \in P, \forall n \in \mathbb{N}, k \neq n \times p$, where $P$ is the list of already found periods.

The list of such $k$ values constitute the set of found seasonality periods $P$. To understand this further, consider this hypothetical time series of hourly frequency that has clear weekly seasonality below

[![Time series with a weekly seasonality](https://raw.githubusercontent.com/iskandergaba/auto-period-finder/master/assets/images/timeseries.png)](https://raw.githubusercontent.com/iskandergaba/auto-period-finder/master/assets/images/timeseries.png)

Now let's look at the corresponding ACF for the time series above:

[![Autocorrelation function of a time series with a weekly seasonality](https://raw.githubusercontent.com/iskandergaba/auto-period-finder/master/assets/images/acf.png)](https://raw.githubusercontent.com/iskandergaba/auto-period-finder/master/assets/images/acf.png)

You can see that the autocorrelation coefficient for lag value 168 hours (i.e. one week) is a local maximum (red-border square). Similarly, autocorrelation coefficient for lag values that are multiples of 168 (gray-border squares). We can therefore conclude that this time series has a weekly seasonality period.

### Notes
- The first condition is needed because a seasonality period cannot neither be 1 (a trivial case), nor greater than half the length of the target time series (by definition, a seasonality has to manifest itself at least twice in a given time series).
- The third condition favors eliminating redundant seasonality periods that are multiples of each others. The algorithm does allow, however, finding seasonality periods that divide already found seasonality periods.
- The periods detection uses `argmax` on the ACF to select seasonality period candidates before checking they satisfy the conditions discussed above. Therefore, the list of seasonality periods are returned in the descending order of their corresponding ACF coefficients.

## References
- [1] Hyndman, R.J., & Athanasopoulos, G. (2021) Forecasting: principles and practice, 3rd edition, OTexts: Melbourne, Australia. [OTexts.com/fpp3](https://otexts.com/fpp3). Accessed on 12-25-2023.
