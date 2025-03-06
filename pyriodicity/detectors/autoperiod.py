from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import argrelmax, detrend, periodogram

from pyriodicity.tools import acf, apply_window, power_threshold, to_1d_array


class Autoperiod:
    """
    Autoperiod periodicity detector.

    Find the periods in a given signal or series using Autoperiod [1]_.

    See Also
    --------
    pyriodicity.CFDAutoperiod
        CFD-Autoperiod periodicity detector.

    References
    ----------
    .. [1] Vlachos, M., Yu, P., & Castelli, V. (2005).
       On periodicity detection and Structural Periodic similarity.
       Proceedings of the 2005 SIAM International Conference on Data Mining.
       https://doi.org/10.1137/1.9781611972757.40

    Examples
    --------
    Start by loading Mauna Loa Weekly Atmospheric CO2 Data from
    `statsmodels <https://statsmodels.org>`_ and downsampling its data to a monthly
    frequency.

    >>> from statsmodels.datasets import co2
    >>> data = co2.load().data
    >>> data = data.resample("ME").mean().ffill()

    Use ``Autoperiod`` to find the list of periods in the data.

    >>> from pyriodicity import Autoperiod
    >>> Autoperiod.detect(data)
    array([12])

    You can specify a lower percentile value should you wish for
    a more lenient detection

    >>> Autoperiod.detect(data, percentile=90)
    array([12])

    You can also increase the number of random data permutations
    for a more robust power threshold estimation

    >>> Autoperiod.detect(data, k=300)
    array([12])

    ``Autoperiod`` is generally a quite robust periodicity detection method.
    The detection algorithm found exactly one periodicity length of 12, suggesting
    a strong yearly periodicity.
    """

    @staticmethod
    def detect(
        data: ArrayLike,
        k: int = 100,
        percentile: int = 95,
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
        k : int, optional, default = 100
            The number of times the data is randomly permuted while estimating the
            power threshold.
        percentile : int, optional, default = 95
            Percentage for the percentile parameter used in computing the power
            threshold. Value must be between 0 and 100 inclusive.
        detrend_func : {'constant', 'linear'}, optional, default = 'linear'
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

        def is_hint_valid(
            x: NDArray,
            hint: float,
            correlation_func: Literal["pearson", "spearman", "kendall"],
        ) -> bool:
            """
            Validate the period hint.

            Parameters
            ----------
            x : array_like
                Data to be investigated. Must be squeezable to 1-d.
            hint : float
                The period hint to be validated.
            correlation_func : {'pearson', 'spearman', 'kendall'}
                The correlation function to be used to calculate the ACF of the series
                or the signal.

            Returns
            -------
            bool
                Whether the period hint is valid.
            """

            def split(
                x: NDArray, y: NDArray, start: int, end: int, split: int
            ) -> tuple:
                """
                Approximate a function at [start, end] with two line segments at
                [start, split] and [split, end].

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
                    Calculate a linear least-squares regression for two sets of
                    measurements.

                Returns
                -------
                numpy.polynomial.Polynomial
                    The first line segment.
                numpy.polynomial.Polynomial
                    The second line segment.
                float
                    The approximation error.
                """
                if not start < split < end:
                    raise ValueError(
                        "Invalid start, split, and end values ({}, {}, {})".format(
                            start, split, end
                        )
                    )
                x1, y1, x2, y2 = (
                    x[start : split + 1],
                    y[start : split + 1],
                    x[split:end],
                    y[split:end],
                )
                line1, stats1 = np.polynomial.Polynomial.fit(x1, y1, deg=1, full=True)
                line2, stats2 = np.polynomial.Polynomial.fit(x2, y2, deg=1, full=True)
                resid1 = (
                    stats1[0][0]
                    if isinstance(stats1, tuple)
                    and len(stats1) > 0
                    and isinstance(stats1[0], np.ndarray)
                    else 0
                )
                resid2 = (
                    stats2[0][0]
                    if isinstance(stats2, tuple)
                    and len(stats2) > 0
                    and isinstance(stats2[0], np.ndarray)
                    else 0
                )
                return line1.convert(), line2.convert(), resid1 + resid2

            length = len(x)
            hint_range = np.arange(
                np.floor((hint + length / (length / hint + 1)) / 2 - 1),
                np.ceil((hint + length / (length / hint - 1)) / 2 + 1),
                dtype=int,
            )
            acf_arr = acf(
                x,
                lag_start=hint_range[0],
                lag_stop=hint_range[-1],
                correlation_func=correlation_func,
            )
            splits = [
                split(hint_range, acf_arr, 0, len(hint_range), i)
                for i in range(1, len(hint_range) - 1)
            ]
            line1, line2, _ = splits[
                np.array([error for _, _, error in splits]).argmin()
            ]
            return line1.coef[-1] > 0 > line2.coef[-1]

        x = to_1d_array(data)

        # Detrend data
        x = x if detrend_func is None else detrend(x, type=detrend_func)
        # Apply window on data
        x = x if window_func is None else apply_window(x, window_func)

        # Compute the power threshold
        detrend_func = "linear" if detrend_func is None else detrend_func
        p_threshold = power_threshold(x, detrend_func, k, percentile)

        # Find period hints
        freq, power = periodogram(x)
        hints = np.array(
            [
                1 / f
                for f, p in zip(freq, power)
                if f >= 1 / len(freq) and p >= p_threshold
            ]
        )

        # Validate period hints
        valid_hints = [h for h in hints if is_hint_valid(x, h, correlation_func)]

        # Return the closest ACF peak to each valid period hint
        length = len(x)
        hint_ranges = [
            np.arange(
                np.floor((h + length / (length / h + 1)) / 2 - 1),
                np.ceil((h + length / (length / h - 1)) / 2 + 1),
                dtype=int,
            )
            for h in valid_hints
        ]
        acf_arrays = [
            acf(
                x,
                lag_start=r[0],
                lag_stop=r[-1],
                correlation_func=correlation_func,
            )
            for r in hint_ranges
        ]
        return np.array(
            list(
                {
                    r[0] + min(argrelmax(arr)[0], key=lambda x: abs(x - h))
                    for h, r, arr in zip(valid_hints, hint_ranges, acf_arrays)
                }
            )
        )
