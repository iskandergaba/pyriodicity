from typing import Callable, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import argrelmax, butter, detrend, sosfiltfilt, periodogram

from pyriodicity.tools import acf, apply_window, to_1d_array


class CFDAutoperiod:
    """
    CFDAutoperiod periodicity detector.

    Find the periods in a given signal or series using CFDAutoperiod [1]_.

    Parameters
    ----------
    endog : array_like
        Data to be investigated. Must be squeezable to 1-d.

    References
    ----------
    .. [1] Puech, T., Boussard, M., D'Amato, A., & Millerand, G. (2020).
       A fully automated periodicity detection in time series. In Advanced
       Analytics and Learning on Temporal Data: 4th ECML PKDD Workshop, AALTD 2019,
       Würzburg, Germany, September 20, 2019, Revised Selected Papers 4 (pp. 43-54).
       Springer International Publishing. https://doi.org/10.1007/978-3-030-39098-3_4

    Examples
    --------
    Start by loading a timeseries datasets and resampling to an appropriate
    frequency.

    >>> from statsmodels.datasets import co2
    >>> data = co2.load().data
    >>> data = data.resample("ME").mean().ffill()

    Use ``CFDAutoperiod`` to find the list of periods in the data.

    >>> from pyriodicity import CFDAutoperiod
    >>> cfd_autoperiod = CFDAutoperiod(data)
    >>> cfd_autoperiod.fit()
    array([12])

    You can specify a lower percentile value should you wish for
    a more lenient detection

    >>> cfd_autoperiod.fit(percentile=90)
    array([12])

    You can also increase the number of random data permutations
    for a more robust power threshold estimation

    >>> cfd_autoperiod.fit(k=300)
    array([12])

    ``CFDAutoperiod`` is considered a more robust variant of ``Autoperiod``.
    The detection algorithm found exactly one periodicity of 12, suggesting
    a strong yearly periodicity.
    """

    def __init__(self, endog: ArrayLike):
        self.y = to_1d_array(endog)

    def fit(
        self,
        k: int = 100,
        percentile: int = 99,
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
        # Detrend data
        self.y = (
            detrend(self.y, type="linear")
            if detrend_func is None
            else detrend(self.y, type=detrend_func)
        )
        # Apply window on data
        self.y = self.y if window_func is None else apply_window(self.y, window_func)

        # Compute the power threshold
        p_threshold = self._power_threshold(self.y, detrend_func, k, percentile)

        # Find period hints
        freq, power = periodogram(self.y, detrend=detrend_func)
        period_hints = np.array(
            [
                1 / f
                for f, p in zip(freq, power)
                if f >= 1 / len(freq) and p >= p_threshold
            ]
        )

        # Replace period hints with their density clustering centroids
        period_hints = self._cluster_period_hints(period_hints, len(self.y))

        # Validate period hints
        period_hints = np.array(
            [
                hint
                for hint in period_hints
                if self._is_hint_valid(
                    self.y,
                    hint,
                    detrend_func=detrend_func,
                    correlation_func=correlation_func,
                )
            ]
        )

        # Return the closest ACF peak to each valid period hint
        acf_arr = acf(self.y, nlags=len(self.y), correlation_func=correlation_func)
        local_argmax = argrelmax(acf_arr)[0]
        return np.array(
            list({min(local_argmax, key=lambda x: abs(x - p)) for p in period_hints})
        )

    @staticmethod
    def _power_threshold(
        y: ArrayLike,
        detrend_func: str,
        k: int,
        p: int,
    ) -> float:
        """
        Compute the power threshold as the p-th percentile of the maximum
        power values of the periodogram of k permutations of the data.

        Parameters
        ----------
        y : array_like
            Data to be investigated. Must be squeezable to 1-d.
        detrend_func : str, default = 'linear'
            The kind of detrending to be applied on the series. It can either be
            'linear' or 'constant'.
        k : int
            The number of times the data is randomly permuted to compute
            the maximum power values.
        p : int
            The percentile value used to compute the power threshold.
            It determines the cutoff point in the sorted list of the maximum
            power values from the periodograms of the permuted data.
            Value must be between 0 and 100 inclusive.

        See Also
        --------
        scipy.signal.periodogram
            Estimate power spectral density using a periodogram.

        Returns
        -------
        float
            Power threshold of the target data.
        """
        max_powers = []
        while len(max_powers) < k:
            _, power_p = periodogram(np.random.permutation(y), detrend=detrend_func)
            max_powers.append(power_p.max())
        max_powers.sort()
        return np.percentile(max_powers, p)

    @staticmethod
    def _cluster_period_hints(period_hints: ArrayLike, n: int) -> NDArray:
        """
        Find the centroids of the period hint density clusters

        Parameters
        ----------
        period_hints : array_like
            List of period hints.
        n : int
            Length of the data.

        Returns
        -------
        NDArray
            List of period hint density cluster centroids.
        """
        hints = np.sort(period_hints)
        eps = [
            hints[i] if i == 0 else 1 + n / (n / hints[i - 1] - 1)
            for i in range(len(hints))
        ]
        clusters = np.split(hints, np.argwhere(hints > eps).flatten())
        return np.array([c.mean() for c in clusters])

    @staticmethod
    def _is_hint_valid(
        y: ArrayLike,
        hint: int,
        detrend_func: Union[str, Callable[[ArrayLike], NDArray]],
        correlation_func: str,
    ) -> bool:
        """
        TODO
        """
        if detrend_func is None:
            detrend_func = "linear"
        if correlation_func is None:
            correlation_func = "pearson"

        # Apply a low pass filter with an adapted cutoff frequency
        length = len(y)
        f_cuttoff = 1 / (length / (length / hint + 1) - 1)
        filter = butter(N=5, Wn=f_cuttoff, output="sos")
        y_filtered = sosfiltfilt(filter, y)

        # Validate the hint
        hint_range = np.arange(hint // 2, 1 + hint + hint // 2, dtype=int)
        acf_arr = acf(y_filtered, nlags=length, correlation_func=correlation_func)
        polynomial = np.polynomial.Polynomial.fit(
            hint_range, detrend(acf_arr[hint_range], type=detrend_func), deg=2
        ).convert()
        derivative = polynomial.deriv()
        return polynomial.coef[-1] < 0 and int(derivative.roots()[0]) in hint_range
