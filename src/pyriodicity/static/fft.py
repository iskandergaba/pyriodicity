from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import detrend

from .._internal.utils import apply_window, to_1d_array


class FFTPeriodicityDetector:
    """
    Fast Fourier Transform (FFT) based periodicity detector.

    Find the periods in a given signal or series using FFT [1]_.

    See Also
    --------
    pyriodicity.OnlineFFTPeriodicityDetector
        Online Fast Fourier Transform (FFT) based periodicity detector.

    References
    ----------
    .. [1] Hyndman, R.J., & Athanasopoulos, G. (2021)
       Forecasting: principles and practice, 3rd edition, OTexts: Melbourne, Australia.
       https://OTexts.com/fpp3/useful-predictors.html#fourier-series.
       Accessed on 09-15-2024.

    Examples
    --------
    Start by loading Mauna Loa Weekly Atmospheric CO2 Data from
    `statsmodels <https://statsmodels.org>`_ and downsampling its data to a monthly
    frequency.

    >>> from statsmodels.datasets import co2
    >>> data = co2.load().data
    >>> data = data.resample("ME").mean().ffill()

    Use ``FFTPeriodicityDetector`` to find the list of periods using FFT, ordered
    by corresponding frequency amplitudes in a descending order.

    >>> from pyriodicity import FFTPeriodicityDetector
    >>> FFTPeriodicityDetector.detect(data)
    array([ 12,   6, 175,  44, 132,  88,  11,  13, 105,  58,  66,  75,  14,
        25,  18,  48,  10,  31,   4,   9,   7,  19,  23,   8,  38,  35,
        40,  28,  20,   3,   5,  15,  29,  22,   2,  24,  53,  33,  26,
        16,  17,  21])

    ``FFTPeriodicityDetector`` tends to be quite sensitive to noise and can find
    many false period lengths. Depending on your data, you can choose to apply
    a window function to get different results. You can also limit the number
    returned period length values to the 3 most signficant ones.

    >>> FFTPeriodicityDetector.detect(window_func="blackman", max_period_count=3)
    array([12, 13, 11])

    As you can see, results are concentrated around the period length value 12,
    indicating a yearly periodicity.
    """

    @staticmethod
    def detect(
        data: ArrayLike,
        window_func: float | str | tuple = "boxcar",
        detrend_func: Literal["constant", "linear"] | None = "linear",
        max_period_count: int | None = None,
    ) -> NDArray:
        """
        Find periods in the given series.

        Parameters
        ----------
        data : array_like
            Data to be investigated. Must be squeezable to 1-d.
        max_period_count : int, optional, default = None
            Maximum number of periods to look for.
        detrend_func : {'constant', 'linear'}, optional, default = 'linear'
            The kind of detrending to be applied on the signal. If None, no detrending
            is applied.
        window_func : float, str, tuple, optional, default = None
            Window function to be applied to the time series. Check
            ``window`` parameter documentation for ``scipy.signal.get_window``
            function for more information on the accepted formats of this
            parameter.

        Returns
        -------
        numpy.ndarray
            Array of detected periods sorted by their amplitude strength in
            descending order. Only unique periods are returned, limited by
            max_period_count if specified. Each period represents the length
            (in samples) of a detected periodicity.

        See Also
        --------
        numpy.fft
            Discrete Fourier Transform.
        scipy.signal.detrend
            Remove linear trend along axis from data.
        scipy.signal.get_window
            Return a window of a given length and type.
        """

        x = to_1d_array(data)

        # Detrend data
        x = x if detrend_func is None else detrend(x, type=detrend_func)

        # Apply the window function on the data
        x = apply_window(x, window_func=window_func)

        # Compute DFT and exclude the DC frequency
        freqs = np.fft.rfftfreq(len(x))[1:]
        ft = np.fft.rfft(x)[1:]

        # Compute period lengths and their respective amplitudes
        periods = np.round(1 / freqs).astype(int)
        amps = abs(ft)

        # A period cannot be greater than half the length of the series
        period_filter = periods < len(x) // 2
        periods = periods[period_filter]
        amps = amps[period_filter]

        # Sort period length values in the descending order of their amplitudes
        periods = periods[np.argsort(-amps)]

        # Return unique period length values
        _, unique_indices = np.unique(periods, return_index=True)
        return periods[np.sort(unique_indices)][:max_period_count]
