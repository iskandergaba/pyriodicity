from collections import Counter
from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import detrend
from scipy.stats import gaussian_kde, zscore

from .._internal.utils import acf, apply_window, to_1d_array


class SAZED:
    """
    Spectral Autocorrelation Zero Ensemble Detector (SAZED).

    Find the periods in a given signal or series using SAZED ensemble method [1]_.

    References
    ----------
    .. [1] Toller, M., Santos, T., & Kern, R. (2019). SAZED: parameter-free
       domain-agnostic season length estimation in time series data. Data Mining
       and Knowledge Discovery, 33(6), 1775-1798.

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
    12

    You can also use the majority voting method:

    >>> SAZED.detect(data, method="majority")
    12
    """

    @staticmethod
    def detect(
        data: ArrayLike,
        window_func: Union[str, float, tuple] = "boxcar",
        detrend_func: Optional[Literal["constant", "linear"]] = "linear",
        method: Literal["optimal", "majority"] = "optimal",
    ) -> Optional[int]:
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

        def s(data: NDArray) -> Optional[int]:
            """
            Spectral component of SAZED (S).

            Parameters
            ----------
            data : NDArray
                Data to be investigated. Must be 1-dimensional.

            Returns
            -------
            Optional[int]
                The detected period length in samples, or None if no valid period
                is found.
            """
            freqs = np.fft.rfftfreq(len(data))[1:]
            ft = np.fft.rfft(data)[1:]

            # Compute period lengths and their respective amplitudes
            periods = np.round(1 / freqs).astype(int)
            amps = abs(ft)

            # Filter periods greater than half the length
            period_filter = periods < len(data) // 2
            periods = periods[period_filter]
            amps = amps[period_filter]

            # Return the period with maximum amplitude
            if len(periods) == 0:
                return None
            return periods[np.argmax(amps)]

        def ze(data: NDArray) -> Optional[int]:
            """
            Zero-crossing mean component (ZE).

            Parameters
            ----------
            data : NDArray
                Data to be investigated. Must be 1-dimensional.

            Returns
            -------
            Optional[int]
                The detected period length in samples based on mean zero-crossing
                distance, or None if no valid period is found.
            """
            signs = np.sign(data)
            signs[signs == 0] = -1

            # Find zero crossings
            zero_crosses = np.where(signs[1:] != signs[:-1])[0]

            if len(zero_crosses) < 2:
                return None

            # Calculate distances between zero crossings
            distances = np.diff(zero_crosses)
            if len(distances) == 0:
                return None

            return int(round(np.mean(distances))) * 2

        def zed(data: NDArray) -> Optional[int]:
            """
            Zero-crossing Density component (ZED).

            Parameters
            ----------
            data : NDArray
                Data to be investigated. Must be 1-dimensional.

            Returns
            -------
            Optional[int]
                The detected period length in samples based on zero-crossing
                density estimation, or None if no valid period is found.
            """
            signs = np.sign(data)
            signs[signs == 0] = -1

            # Find zero crossings
            zero_crosses = np.where(signs[1:] != signs[:-1])[0]

            if len(zero_crosses) < 2:
                return None

            # Calculate distances between zero crossings
            distances = np.diff(zero_crosses)
            if len(distances) == 0:
                return None

            # Use kernel density estimation to find the most common distance
            kde = gaussian_kde(distances)
            x = np.linspace(min(distances), max(distances), 100)
            period = x[np.argmax(kde(x))]

            return round(period * 2)  # Multiply by 2 to get full period

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
