from typing import Literal, Optional, Union

from numpy.typing import ArrayLike, NDArray
from scipy.signal import argrelmax, detrend

from pyriodicity.tools import acf, apply_window, to_1d_array


class ACFPeriodicityDetector:
    """
    Autocorrelation function (ACF) based periodicity detector.

    Find the periods in a given signal or series using its ACF. A lag value
    is considered a period if it is a local maximum of the ACF [1]_.

    References
    ----------
    .. [1] Hyndman, R.J., & Athanasopoulos, G. (2021)
       Forecasting: principles and practice, 3rd edition, OTexts: Melbourne, Australia.
       https://OTexts.com/fpp3/acf.html. Accessed on 09-15-2024.

    Examples
    --------
    Start by loading Mauna Loa Weekly Atmospheric CO2 Data from
    `statsmodels <https://statsmodels.org>`_ and downsampling its data to a monthly
    frequency.

    >>> from statsmodels.datasets import co2
    >>> data = co2.load().data
    >>> data = data.resample("ME").mean().ffill()

    Use ACFPeriodicityDetector to find the list of seasonality periods using the ACF.

    >>> from pyriodicity import ACFPeriodicityDetector
    >>> ACFPeriodicityDetector.detect(data)
    array([ 12,  24,  36,  48,  60,  72,  84,  96, 108, 120, 132, 143, 155,
       167, 179, 191, 203, 215, 227, 239, 251])

    You can use a different correlation function like Spearman

    >>> ACFPeriodicityDetector.detect(data, correlation_func="spearman")
    array([ 12,  24,  36,  48,  60,  72,  84,  96, 108, 120, 132, 143, 155,
       167, 179, 191, 203, 215, 227, 239, 251])

    All of the returned values are either multiples of 12 or very close to it,
    suggesting a clear yearly periodicity.
    You can also get the most prominent period length value by setting
    ``max_period_count`` to 1.

    >>> ACFPeriodicityDetector.detect(data, max_period_count=1)
    array([12])
    """

    @staticmethod
    def detect(
        data: ArrayLike,
        max_period_count: Optional[int] = None,
        detrend_func: Optional[Literal["constant", "linear"]] = "linear",
        window_func: Optional[Union[str, float, tuple]] = None,
        correlation_func: Literal["pearson", "spearman", "kendall"] = "pearson",
    ) -> NDArray:
        """
        Find periods in the given series.

        Parameters
        ----------
        data : array_like
            Data to be investigated. Must be squeezable to 1-d.
        max_period_count : int, optional, default = None
            Maximum number of periods to look for.
        detrend_func : {'constant', 'linear'} or None, default = 'linear'
            The kind of detrending to be applied on the signal. If None, no detrending
            is applied.
        window_func : float, str, tuple, optional, default = None
            Window function to be applied to the time series. Check
            ``window`` parameter documentation for ``scipy.signal.get_window``
            function for more information on the accepted formats of this
            parameter.
        correlation_func : {'pearson', 'spearman', 'kendall'}, default = 'pearson'
            The correlation function to be used to calculate the ACF of the signal.

        Returns
        -------
        NDArray
            List of detected periods.

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
        """
        x = to_1d_array(data)

        # Detrend data
        x = x if detrend_func is None else detrend(x, type=detrend_func)

        # Apply window on data
        x = x if window_func is None else apply_window(x, window_func)

        # Compute the ACF
        acf_arr = acf(
            x,
            lag_start=0,
            lag_stop=len(x) // 2,
            correlation_func=correlation_func,
        )

        # Find the local argmax of the first half of the ACF array
        local_argmax = argrelmax(acf_arr)[0]

        # Argsort the local maxima in the ACF array in a descending order
        periods = local_argmax[acf_arr[local_argmax].argsort()][::-1]

        # Return the requested maximum count of detected periods
        return periods[:max_period_count]
