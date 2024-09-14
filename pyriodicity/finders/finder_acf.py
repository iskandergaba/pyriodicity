from typing import Callable, Optional, Union

from numpy.typing import ArrayLike, NDArray
from scipy.signal import argrelmax

from pyriodicity.tools import acf, apply_window, detrend, to_1d_array


class AutocorrelationPeriodFinder:
    """
    Autocorrelation function (ACF) based seasonality periods automatic finder.

    Find the periods of a given time series using its ACF. A time delta
    is considered a period if it is a local maximum of ACF.

    Parameters
    ----------
    endog : array_like
        Data to be investigated. Must be squeezable to 1-d.

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

    >>> data = data.resample("ME").mean().ffill()

    Use AutocorrelationPeriodFinder to find the list of seasonality periods based on
    ACF.

    >>> period_finder = AutocorrelationPeriodFinder(data)
    >>> periods = period_finder.fit()

    You can get the most prominent period by setting max_period_count to 1

    >>> period_finder.fit(max_period_count=1)

    You can also use a different correlation function like Spearman

    >>> period_finder.fit(correlation_func="spearman")
    """

    def __init__(self, endog: ArrayLike):
        self.y = to_1d_array(endog)

    def fit(
        self,
        max_period_count: Optional[int] = None,
        detrend_func: Optional[Union[str, Callable[[ArrayLike], NDArray]]] = "linear",
        window_func: Optional[Union[str, float, tuple]] = None,
        correlation_func: Optional[str] = "pearson",
    ) -> NDArray:
        """
        Find seasonality periods of the given time series automatically.

        Parameters
        ----------
        max_period_count : int, optional, default = None
            Maximum number of periods to look for.
        detrend_func : str, callable, default = None
            The kind of detrending to be applied on the series. It can either be
            'linear' or 'constant' if it the parameter is of 'str' type, or a
            custom function that returns a detrended series.
        window_func : float, str, tuple optional, default = None
            Window function to be applied to the time series. Check
            'window' parameter documentation for scipy.signal.get_window
            function for more information on the accepted formats of this
            parameter.
        correlation_func : str, default = 'pearson'
            The correlation function to be used to calculate the ACF of the time
            series. Possible values are ['pearson', 'spearman', 'kendall'].

        See Also
        --------
        scipy.signal.detrend
            Remove linear trend along axis from data.
        scipy.signal.get_window
            Return a window of a given length and type.
        scipy.stats.kendalltau
            Calculate Kendall's tau, a correlation measure for ordinal data.
        scipy.stats.pearsonr
            Pearson correlation coefficient and p-value for testing non-correlation.
        scipy.stats.spearmanr
            Calculate a Spearman correlation coefficient with associated p-value.


        Returns
        -------
        NDArray
            List of detected seasonality periods.
        """
        return self.__find_periods(
            max_period_count, detrend_func, window_func, correlation_func
        )

    def __find_periods(
        self,
        max_period_count: Optional[int],
        detrend_func: Optional[Union[str, Callable[[ArrayLike], NDArray]]] = "linear",
        window_func: Optional[Union[str, float, tuple]] = None,
        correlation_func: Optional[str] = "pearson",
    ) -> NDArray:

        # Detrend data
        self.y = self.y if detrend_func is None else detrend(self.y, detrend_func)

        # Apply window on data
        self.y = self.y if window_func is None else apply_window(self.y, window_func)

        # Compute the ACF
        acf_arr = acf(self.y, len(self.y) // 2, correlation_func)

        # Find the local argmax of the first half of the ACF array
        local_argmax = argrelmax(acf_arr)[0]

        # Argsort the local maxima in the ACF array in a descending order
        periods = local_argmax[acf_arr[local_argmax].argsort()][::-1]

        # Return the requested maximum count of detected periods
        return periods[:max_period_count]
