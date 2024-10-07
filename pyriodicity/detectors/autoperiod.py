from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import argrelmax, detrend, periodogram

from pyriodicity.tools import acf, apply_window, power_threshold, to_1d_array


class Autoperiod:
    """
    Autoperiod periodicity detector.

    Find the periods in a given signal or series using Autoperiod.

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
    Start by loading a timeseries dataset.

    >>> from statsmodels.datasets import co2
    >>> data = co2.load().data

    You can resample the data to whatever frequency you want.

    >>> data = data.resample("ME").mean().ffill()

    Use Autoperiod to find the list of periods in the data.

    >>> autoperiod = Autoperiod(data)
    >>> periods = autoperiod.fit()

    You can specify a lower percentile value for a more lenient detection

    >>> autoperiod.fit(percentile=90)

    Or increase the number of random data permutations for a better power threshold estimation

    >>> autoperiod.fit(k=300)
    """

    def __init__(self, endog: ArrayLike):
        self.y = to_1d_array(endog)

    def fit(
        self,
        k: int = 100,
        percentile: int = 95,
        detrend_func: Optional[str] = "linear",
        window_func: Optional[Union[str, float, tuple]] = None,
        correlation_func: Optional[str] = "pearson",
    ) -> NDArray:
        """
        Find periods in the given series.

        Parameters
        ----------
        k : int, optional, default = 100
            The number of times the data is randomly permuted while estimating the
            power threshold.
        percentile : int, optional, default = 95
            Percentage for the percentile parameter used in computing the power
            threshold. Value must be between 0 and 100 inclusive.
        detrend_func : str, default = 'linear'
            The kind of detrending to be applied on the signal. It can either be
            'linear' or 'constant'.
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
            List of detected periods.
        """
        # Detrend data
        self.y = self.y if detrend_func is None else detrend(self.y, type=detrend_func)
        # Apply window on data
        self.y = self.y if window_func is None else apply_window(self.y, window_func)

        # Compute the power threshold
        p_threshold = power_threshold(self.y, detrend_func, k, percentile)

        # Find period hints
        freq, power = periodogram(self.y, window=None, detrend=False)
        hints = np.array(
            [
                1 / f
                for f, p in zip(freq, power)
                if f >= 1 / len(freq) and p >= p_threshold
            ]
        )

        # Compute the ACF
        length = len(self.y)
        acf_arr = acf(
            self.y, lag_start=0, lag_stop=length, correlation_func=correlation_func
        )

        # Validate period hints
        valid_hints = []
        for p in hints:
            q = length / p
            start = np.floor((p + length / (q + 1)) / 2 - 1).astype(int)
            end = np.ceil((p + length / (q - 1)) / 2 + 1).astype(int)

            splits = [
                self._split(np.arange(len(acf_arr)), acf_arr, start, end, i)
                for i in range(start + 2, end)
            ]
            line1, line2, _ = splits[
                np.array([error for _, _, error in splits]).argmin()
            ]

            if line1.coef[-1] > 0 > line2.coef[-1]:
                valid_hints.append(p)

        # Return the closest ACF peak to each valid period hint
        local_argmax = argrelmax(acf_arr)[0]
        return np.array(
            list({min(local_argmax, key=lambda x: abs(x - p)) for p in valid_hints})
        )

    @staticmethod
    def _split(x: ArrayLike, y: ArrayLike, start: int, end: int, split: int) -> tuple:
        """
        Approximate a function at [start, end] with two line segments at
        [start, split - 1] and [split, end].

        Parameters
        ----------
        x : array_like
            The x-coordinates of the data points.
        y : array_like
            The y-coordinates of the data points.
        start : int
            The start index of the data points to be approximated.
        end : int
            The end index of the data points to be approximated.
        split : int
            The split index of the data points to be approximated.

        See Also
        --------
        scipy.stats.linregress
            Calculate a linear least-squares regression for two sets of measurements.

        Returns
        -------
        numpy.polynomial.Polynomial
            The first line segment.
        numpy.polynomial.Polynomial
            The second line segment.
        float
            The approximation error.
        """
        x1, y1, x2, y2 = (
            x[start:split],
            y[start:split],
            x[split : end + 1],
            y[split : end + 1],
        )
        line1, stats1 = np.polynomial.Polynomial.fit(x1, y1, deg=1, full=True)
        line2, stats2 = np.polynomial.Polynomial.fit(x2, y2, deg=1, full=True)
        resid1 = 0 if len(stats1[0]) == 0 else stats1[0][0]
        resid2 = 0 if len(stats2[0]) == 0 else stats2[0][0]
        return line1.convert(), line2.convert(), resid1 + resid2
