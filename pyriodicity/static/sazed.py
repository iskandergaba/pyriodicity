"""
SAZED (Spectral Autocorrelation Zero Ensemble Detector).

Find the periods in a given signal or series using SAZED.

References
----------
.. [1] Gaba, Iskandar. (2023) SazedR: A package for estimating the season length
   of a seasonal time series.
   https://github.com/cran/sazedR
"""

from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import detrend
from scipy.stats import gaussian_kde

from .._internal.utils import acf, apply_window, to_1d_array


class SAZED:
    """
    SAZED (Spectral Autocorrelation Zero Ensemble Detector).

    Find the periods in a given signal or series using SAZED ensemble method [1]_.

    See Also
    --------
    pyriodicity.FFTPeriodicityDetector
        Fast Fourier Transform (FFT) based periodicity detector.
    pyriodicity.ACFPeriodicityDetector
        Autocorrelation Function (ACF) based periodicity detector.

    References
    ----------
    .. [1] Gaba, Iskandar. (2023) SazedR: A package for estimating the season length
       of a seasonal time series.
       https://github.com/cran/sazedR

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
    def _spectral(data: NDArray) -> Optional[int]:
        """Spectral component of SAZED (S)."""
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

    @staticmethod
    def _ze(data: NDArray) -> Optional[int]:
        """Zero-crossing mean component (ZE)."""
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

    @staticmethod
    def _zed(data: NDArray) -> Optional[int]:
        """Zero-crossing Density component (ZED)."""
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

    @staticmethod
    def _detect_optimal(data: NDArray) -> Optional[int]:
        """Optimal SAZED detection method."""
        # Compute ACF once
        acf_data = acf(data)

        # Get all estimates directly
        all_periods = [
            SAZED._spectral(data),
            SAZED._ze(data),
            SAZED._zed(data),
            SAZED._spectral(acf_data),
            SAZED._ze(acf_data),
            SAZED._zed(acf_data),
        ]

        # Filter None values and get periods > 2
        valid_periods = [p for p in all_periods if p is not None and p > 2]

        if not valid_periods:
            return None

        unique_periods = np.unique(valid_periods)

        # Compute certainty for each period
        certainties = []
        for period in unique_periods:
            if period <= 2 or len(data) // period <= 3:
                certainties.append(-1)
                continue

            # Split data into segments of length period
            n_segments = len(data) // period
            segments = np.array(
                [data[i * period : (i + 1) * period] for i in range(n_segments)]
            )

            # Compute correlation matrix and get minimum correlation
            corr_matrix = np.corrcoef(segments)
            certainties.append(np.min(corr_matrix))

        # Return period with highest certainty
        return unique_periods[np.argmax(certainties)]

    @staticmethod
    def _detect_majority(data: NDArray) -> Optional[int]:
        """Majority voting SAZED detection method."""
        # Compute ACF once
        acf_data = acf(data)

        # Get all estimates directly
        all_periods = [
            SAZED._spectral(data),
            SAZED._ze(data),
            SAZED._zed(data),
            SAZED._spectral(acf_data),
            SAZED._ze(acf_data),
            SAZED._zed(acf_data),
        ]

        # Filter None values and get periods > 2
        valid_periods = [p for p in all_periods if p is not None and p > 2]

        if not valid_periods:
            return None

        # Count occurrences
        unique_periods = np.unique(valid_periods)
        period_counts = {p: valid_periods.count(p) for p in unique_periods}

        # Get the period(s) with maximum count
        max_count = max(period_counts.values())
        max_periods = [p for p, count in period_counts.items() if count == max_count]

        if len(max_periods) == 1:
            # Clear winner
            return max_periods[0]
        else:
            # Tie - take the largest period following R implementation
            return max(max_periods)

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
            Input data array.
        window_func : str or float or tuple, optional
            Window function to apply to the data. See scipy.signal.get_window for
            details. Default is "boxcar".
        detrend_func : {"linear", "constant"} or None, optional
            Detrending function to apply to the data. If None, no detrending is
            performed. Default is "linear".
        method : {"optimal", "majority"}, optional
            The ensemble method to use. 'optimal' uses correlation-based period
            selection (equivalent to R's sazed()), while 'majority' uses voting
            (equivalent to R's sazed.maj()). Default is "optimal".

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
            If method is not "optimal" or "majority".
        """
        if (
            np.any(np.isnan(data))
            or np.any(np.isinf(data))
            or np.any(np.iscomplex(data))
        ):
            return None

        x = to_1d_array(data)

        if len(x) < 4 or np.var(x) == 0:
            return None

        # Data preprocessing
        x = x if detrend_func is None else detrend(x, type=detrend_func)
        x = apply_window(x, window_func)
        x = (x - np.mean(x)) / np.std(x)  # z-normalize

        # Choose detection method
        if method == "optimal":
            return SAZED._detect_optimal(x)
        elif method == "majority":
            return SAZED._detect_majority(x)
        else:
            raise ValueError('method must be either "optimal" or "majority"')
