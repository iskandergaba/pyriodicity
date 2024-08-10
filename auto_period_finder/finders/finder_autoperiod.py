from typing import Callable, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import argrelmax, periodogram
from scipy.stats import linregress

from auto_period_finder.tools import acf, apply_window, detrend, to_1d_array


class AutoperiodDetector:
    """
    Autoperiod periodicity detector.

    Find the seasonlity periods of a given time series using Autoperiod.

    Parameters
    ----------
    endog : array_like
        Data to be investigated. Must be squeezable to 1-d.

    References
    ----------
    .. [1] Vlachos, M., Yu, P., & Castelli, V. (2005).
    On periodicity detection and Structural Periodic similarity.
    Proceedings of the 2005 SIAM International Conference on Data Mining.
    https://doi.org/10.1137/1.9781611972757.40

    Examples
    --------
    Start by loading a timeseries dataset with a frequency.

    >>> from statsmodels.datasets import co2
    >>> data = co2.load().data

    You can resample the data to whatever frequency you want.

    >>> data = data.resample("ME").mean().ffill()

    Use AutoperiodDetector to find the list of seasonality periods based on
    ACF.

    >>> autoperiod_detector = AutoperiodDetector(data)
    >>> periods = autoperiod_detector.fit()
    """

    def __init__(self, endog: ArrayLike):
        self.y = to_1d_array(endog)

    def fit(
        self,
        k: int = 300,
        percentile: int = 95,
        detrend_func: Optional[Union[str, Callable[[ArrayLike], NDArray]]] = "linear",
        window_func: Optional[Union[str, float, tuple]] = None,
        correlation_func: Optional[str] = "pearson",
    ) -> NDArray:
        """
        Detect the seasonality periods of the given time series.

        Parameters
        ----------
        k : int, optional, default = 300
            TODO explanation.
        percentile : int, optional, default = 95
            TODO explanation.
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
        # Detrend data
        self.y = self.y if detrend_func is None else detrend(self.y, detrend_func)
        # Apply window on data
        self.y = self.y if window_func is None else apply_window(self.y, window_func)

        # Compute the power threshold
        p_threshold = self._power_threshold(self.y, k, percentile)

        # Find period hints
        freq, power = periodogram(self.y, window=None, detrend=None)
        period_hints = np.array(
            [
                1 / f
                for f, p in zip(freq, power)
                if f >= 1 / len(freq) and p >= p_threshold
            ]
        )

        # Compute the ACF
        length = len(self.y)
        acf_arr = acf(self.y, nlags=length, correlation_func=correlation_func)

        # Validate period hints
        # TODO Improve code
        period_hints_valid = []
        for p in period_hints:
            q = length / p
            start = np.floor((p + length / (q + 1)) / 2 - 1).astype(int)
            end = np.ceil((p + length / (q - 1)) / 2 + 1).astype(int)
            t = (
                start
                + 2
                + np.array(
                    [
                        self._split_error(
                            np.arange(len(acf_arr)), acf_arr, start, end, i
                        )
                        for i in range(start + 2, end)
                    ]
                ).argmin()
            )
            if self._is_split_valid(np.arange(len(acf_arr)), acf_arr, start, end, t):
                period_hints_valid.append(p)

        period_hints_valid = np.array(period_hints_valid)

        # Return the ACF peaks for valid period hints
        local_argmax = argrelmax(acf_arr)[0]
        return np.array(
            list(
                {
                    min(local_argmax, key=lambda x: abs(x - p))
                    for p in period_hints_valid
                }
            )
        )

    ## Compute the power threshold
    # TODO documentation
    @staticmethod
    def _power_threshold(y: ArrayLike, k: int, percentile: int):
        max_powers = []
        while len(max_powers) < k:
            _, power_p = periodogram(
                np.random.permutation(y), window=None, detrend=None
            )
            max_powers.append(power_p.max())
        max_powers.sort()
        return np.percentile(max_powers, percentile)

    ## Compute the split error
    # TODO documentation
    def _split_error(
        self, x: ArrayLike, y: ArrayLike, start: int, end: int, split: int
    ):
        _, _, error = self._split(x, y, start, end, split)
        return error

    ## Check if the split is valid
    # TODO documentation
    def _is_split_valid(
        self, x: ArrayLike, y: ArrayLike, start: int, end: int, split: int
    ):
        line1, line2, _ = self._split(x, y, start, end, split)
        return line1.slope > 0 > line2.slope

    # Approximate a function at [start, end] with two line segments at [start, split - 1] and [split, end]
    def _split(self, x: ArrayLike, y: ArrayLike, start: int, end: int, split: int):
        x1, y1, x2, y2 = (
            x[start:split],
            y[start:split],
            x[split : end + 1],
            y[split : end + 1],
        )
        line1 = linregress(x1, y1)
        line2 = linregress(x2, y2)
        error = np.sum(np.abs(y1 - (line1.intercept + line1.slope * x1))) + np.sum(
            np.abs(y2 - (line2.intercept + line2.slope * x2))
        )
        return line1, line2, error
