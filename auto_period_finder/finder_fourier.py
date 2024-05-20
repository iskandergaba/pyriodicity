from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.signal import get_window
from statsmodels.tools.typing import ArrayLike1D


class FourierPeriodFinder:
    """
    Autocorrelation function (ACF) based seasonality periods automatic finder.

    Find the periods of a given time series using its ACF. A time delta
    is considered a period if:
        1- It is a local maximum of ACF.\n
        2- Its multiples are also local maxima.\n
        3- It is not a multiple of an already discovered period. For example,
        it is redundant to return a 2 year seasonality period if we have already
        found a 1 year period. The inverse, however, is not necessarily true.

    Parameters
    ----------
    endog : array_like
        Data to be investigated. Must be squeezable to 1-d.
    acf_kwargs: dict, optional
        Arguments to pass to the ACF.

    See Also
    --------
    statsmodels.tsa.stattools.acf
        Autocorrelation function.
    statsmodels.tsa.seasonal.STL
        Season-Trend decomposition using LOESS.
    statsmodels.tsa.seasonal.seasonal_decompose
        Seasonal decomposition using moving averages.

    References
    ----------
    .. [1] Hyndman, R.J., & Athanasopoulos, G. (2021)
    Forecasting: principles and practice, 3rd edition, OTexts: Melbourne, Australia.
    OTexts.com/fpp3/stlfeatures.html. Accessed on 12-23-2023.

    Examples
    --------
    Start by loading a timeseries dataset with a frequency.

    >>> from statsmodels.datasets import co2
    >>> data = co2.load().data

    You can resample the data to whatever frequency you want.

    >>> data = data.resample("M").mean().ffill()

    Use AutoPeriodFinder to find the list of seasonality periods based on ACF.

    >>> period_finder = AutoPeriodFinder(data)
    >>> periods = period_finder.fit()

    You can also find the most prominent period either ACF-wise or variance-wise.

    >>> strongest_period_acf = period_finder.fit_find_strongest_acf()
    >>> strongest_period_var = period_finder.fit_find_strongest_var()
    """

    def __init__(self, endog: ArrayLike1D):
        self.y = self.__to_1d_array(endog)

    def fit(
        self,
        max_period_count: Optional[Union[int, None]] = None,
        window_func: Optional[Union[str, float, tuple, None]] = None,
    ) -> list:
        """
        Find seasonality periods of the given time series automatically.

        Parameters
        ----------
        vicinity_radius : int, optional, default = None
            How many data points, before and after, a period candidate
            value to consider for satisfying the periodicity conditions.
            Essentially, the algorithm will verify that at least one point
            in the vicinity (defined by this parameter) of every multiple
            of the candidate value is a local maximum.
            This helps mitigate the effects of the forward and backward
            noise shifts of the period value. It is also effective
            at reducing the number of detected period values that are
            too tightly bunched together.
        max_period_count : int, optional, default = None
            Maximum number of periods to look for.

        Returns
        -------
        list
            List of periods.
        """
        return self.__find_periods(
            self.y,
            max_period_count,
            window_func=window_func,
        )

    def __find_periods(
        self,
        y: ArrayLike1D,
        max_period_count: Optional[Union[int, None]],
        window_func: Optional[Union[str, float, tuple, None]],
    ) -> list:
        y_windowed = (
            y
            if window_func is None
            else self.__get_windowed_y(y, window_func=window_func)
        )

        # TODO: Rename `mags` and add some explanation comments
        ft = np.fft.rfft(y_windowed)[1:]
        mags = abs(ft)
        freqs = np.fft.rfftfreq(len(y_windowed), d=1)[1:]
        periods = np.round(1 / freqs)

        # A period cannot be greater than half the length of the series
        result = pd.Series(index=periods, data=mags)[periods < len(y_windowed) / 2]
        return (
            result.sort_values(ascending=False).index.unique().values[:max_period_count]
        )

    @staticmethod
    def __get_windowed_y(y, window_func) -> np.ndarray:
        return (y - np.median(y)) * get_window(window=window_func, Nx=len(y))

    @staticmethod
    def __to_1d_array(x):
        y = np.ascontiguousarray(np.squeeze(np.asarray(x)), dtype=np.double)
        if y.ndim != 1:
            raise ValueError("y must be a 1d array")
        return y
