from collections import Counter
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.fft import dct
from scipy.optimize import fminbound
from scipy.signal import detrend, periodogram
from scipy.stats import gaussian_kde, zscore

from .._internal.utils import acf, apply_window, to_1d_array


class SAZED:
    """
    Spectral Autocorrelation Zero Ensemble Detector (SAZED).

    Find the periods in a given signal or series using SAZED ensemble method [1]_.

    Notes
    -----
    While the original paper uses the Sheather-Jones bandwidth selector, this
    implementation uses the Improved Sheather-Jones (ISJ) selector [2]_.

    References
    ----------
    .. [1] Toller, M., Santos, T., & Kern, R. (2019). SAZED: parameter-free
       domain-agnostic season length estimation in time series data. Data Mining
       and Knowledge Discovery, 33(6), 1775-1798.
       https://doi.org/10.1007/s10618-019-00645-z
    .. [2] Botev, Z. I., Grotowski, J. F., & Kroese, D. P. (2010). Kernel density
       estimation via diffusion. The Annals of Statistics, 38(5), 2916-2957.
       https://doi.org/10.1214/10-AOS799

    Examples
    --------
    Start by loading Mauna Loa Weekly Atmospheric CO2 Data from
    `statsmodels <https://statsmodels.org>`_ and downsampling its data to a monthly
    frequency.

    >>> from statsmodels.datasets import co2
    >>> data = co2.load().data
    >>> data = data.resample("ME").mean().ffill()

    Use SAZED to find periods using the ensemble method.

    >>> from pyriodicity import SAZED
    >>> SAZED.detect(data)
    np.int64(12)

    You can also use the majority voting method:

    >>> SAZED.detect(data, method="majority")
    np.int64(12)
    """

    @staticmethod
    def detect(
        data: ArrayLike,
        window_func: str | float | tuple = "boxcar",
        detrend_func: Literal["constant", "linear"] | None = "linear",
        method: Literal["optimal", "majority"] = "optimal",
    ) -> int | None:
        """
        Detect a period in the input data using the SAZED ensemble method.

        Parameters
        ----------
        data : array_like
            Data to be investigated. Must be squeezable to 1-d.
        window_func : float, str, tuple, default = 'boxcar'
            Window function to be applied to the time series. Check
            ``window`` parameter documentation for ``scipy.signal.get_window``
            function for more information on the accepted formats of this
            parameter.
        detrend_func : {'constant', 'linear'}, optional, default = 'linear'
            The kind of detrending to be applied on the signal. If None, no detrending
            is applied.
        method : {'optimal', 'majority'}, default = 'optimal'
            The ensemble method to use. 'optimal' uses correlation-based period
            selection, while 'majority' uses voting.

        Returns
        -------
        int or None
            The detected period length in samples, or None if no valid period
            is found.

        See Also
        --------
        scipy.signal.detrend
            Remove linear trend along axis from data.
        scipy.signal.get_window
            Return a window of a given length and type.

        Raises
        ------
        ValueError
            - If input data contains NaN or Inf values
            - If method is neither 'optimal' nor 'majority'
        """

        def s(data: NDArray) -> int | None:
            """
            Spectral component of SAZED (S).

            Parameters
            ----------
            data : NDArray
                Data to be investigated. Must be 1-dimensional.

            Returns
            -------
            int | None
                The detected period length in samples, or None if no valid period
                is found.
            """

            # Compute the periodogram and drop the DC frequency
            freqs, psd = periodogram(data, window="boxcar", detrend=False)
            freqs, psd = freqs[1:], psd[1:]

            # Compute period lengths and their respective amplitudes
            periods = np.round(1 / freqs).astype(int)

            # Filter periods greater than half the length
            period_filter = periods < len(data) // 2
            periods = periods[period_filter]
            psd = psd[period_filter]

            # Return the period with maximum amplitude
            if len(periods) == 0:
                return None
            return periods[np.argmax(psd)]

        def ze(data: NDArray) -> int | None:
            """
            Zero-crossing mean component (ZE).

            Parameters
            ----------
            data : NDArray
                Data to be investigated. Must be 1-dimensional.

            Returns
            -------
            int | None
                The detected period length in samples based on mean zero-crossing
                distance, or None if no valid period is found.
            """

            # Find zero crossings
            zero_crosses = np.where(np.diff(np.signbit(data)))[0]

            # Return None if there are not enough zero crossings
            if len(zero_crosses) < 2:
                return None

            # Calculate distances between zero crossings
            distances = np.diff(zero_crosses)

            # Return the mean distance multiplied by 2 (for full period)
            return (
                None
                if len(distances) == 0
                else np.rint(np.mean(distances)).astype(int) * 2
            )

        def zed(data: NDArray) -> int | None:
            """
            Zero-crossing Density component (ZED).

            Parameters
            ----------
            data : NDArray
                Data to be investigated. Must be 1-dimensional.

            Returns
            -------
            int | None
                The detected period length in samples based on zero-crossing
                density estimation, or None if no valid period is found.
            """

            def isj(kde: gaussian_kde) -> float:
                """
                Improved Sheather-Jones (ISJ) bandwidth selector.

                A ``bw_method`` callable for ``scipy.stats.gaussian_kde``, ported
                from Botev's R reference implementation
                (https://web.maths.unsw.edu.au/~zdravkobotev/kde.R). Returns the
                bandwidth divided by the data standard deviation, which is the scale
                ``factor`` that ``gaussian_kde`` expects.
                """

                def fixed_point(t: float, n: int, i: NDArray, a2: NDArray) -> float:
                    f = (
                        2
                        * np.pi ** (2 * 7)
                        * np.sum(i**7 * a2 * np.exp(-i * np.pi**2 * t))
                    )
                    for s in range(6, 1, -1):
                        k0 = np.prod(np.arange(1, 2 * s, 2)) / np.sqrt(2 * np.pi)
                        const = (1 + 0.5 ** (s + 0.5)) / 3
                        time = (2 * const * k0 / n / f) ** (2 / (3 + 2 * s))
                        f = (
                            2
                            * np.pi ** (2 * s)
                            * np.sum(i**s * a2 * np.exp(-i * np.pi**2 * time))
                        )
                    # Ensure f stays finite
                    f = max(f, np.finfo(float).tiny)
                    return t - (2 * n * np.sqrt(np.pi) * f) ** (-2 / 5)

                data = kde.dataset.ravel()
                n = len(np.unique(data))

                # Bin the data onto a power-of-two mesh over a padded range
                n_grid = 2**14
                minimum, maximum = data.min(), data.max()
                data_range = maximum - minimum
                low, high = minimum - data_range / 2, maximum + data_range / 2
                r = high - low
                hist = np.histogram(data, bins=n_grid, range=(low, high))[0]
                hist = hist / hist.sum()

                # Discrete cosine transform of the binned density (Botev's a2)
                a2 = (dct(hist, type=2, norm=None)[1:] / 2) ** 2
                i = np.arange(1, n_grid, dtype=float) ** 2

                # Solve the fixed-point equation for the smoothing parameter t*
                t_star = fminbound(
                    lambda x: abs(fixed_point(x, n, i, a2)), 0, 0.1, xtol=1e-14
                )

                # Convert the smoothing parameter to a scipy bandwidth factor
                return np.sqrt(t_star) * r / np.std(data, ddof=1)

            # Find zero crossings
            zero_crosses = np.where(np.diff(np.signbit(data)))[0]

            # Return None if there are not enough zero crossings
            if len(zero_crosses) < 2:
                return None

            # Calculate distances between zero crossings
            distances = np.diff(zero_crosses)
            if len(distances) < 2:
                return None

            # Constant distances make the KDE covariance singular; the mode is
            # then simply the repeated distance value itself
            if np.ptp(distances) == 0:
                period = distances[0] * 2
            else:
                # Use kernel density estimation to find the most common distance
                kde = gaussian_kde(distances, bw_method=isj)
                x = np.unique(distances)
                period = x[np.argmax(kde(x))] * 2

            # Return the rounded period length
            return np.rint(period).astype(int)

        # Validate input data
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            raise ValueError("Input data contains NaN or Inf values.")

        # Validate the estimation method
        if method not in ["optimal", "majority"]:
            raise ValueError("Estimation method must be either 'optimal' or 'majority'")

        # Squeeze the data to 1D
        x = to_1d_array(data)

        # Return None if data is constant
        if np.var(x) == 0:
            return None

        # Detrend data
        x = x if detrend_func is None else detrend(x, type=detrend_func)
        # Apply window on data
        x = apply_window(x, window_func)
        # Normalize data
        x = zscore(x)

        # Compute ACF
        acf_arr = acf(x)
        # Compute periodicty length estimates
        period_counter = Counter(
            [s(x), ze(x), zed(x), s(acf_arr), ze(acf_arr), zed(acf_arr)]
        )
        # Drop the None key from the counter if it exists
        del period_counter[None]

        # Drop periods too long to yield at least two comparable segments
        for period in [p for p in period_counter if len(x) // p < 2]:
            del period_counter[period]

        # Return None if no periodicity length estimates are found
        if len(period_counter) == 0:
            return None

        if method == "majority":
            # Return the greatest periodicity length with maximum occurrences
            return max(period_counter, key=lambda k: (period_counter[k], k))
        else:
            # Default to "optimal". Compute periodicity length certainties
            certainties = []
            for period in period_counter:
                # Split data into segments of length period
                n_segments = len(x) // period
                segments = np.array(
                    [x[i * period : (i + 1) * period] for i in range(n_segments)]
                )
                # Compute correlation matrix and get minimum correlation
                corr_matrix = np.corrcoef(segments)
                certainties.append(np.min(corr_matrix))

            # Return period with highest certainty
            return list(period_counter.keys())[np.argmax(certainties)]
