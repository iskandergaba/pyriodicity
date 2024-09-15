from typing import Callable, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyriodicity.tools import apply_window, detrend, to_1d_array


class FFTPeriodicityDetector:
    """
    Fast Fourier Transform (FFT) based periodicity detector.

    Find the periods in a given signal or series using FFT.

    Parameters
    ----------
    endog : array_like
        Data to be investigated. Must be squeezable to 1-d.

    References
    ----------
    .. [1] Hyndman, R.J., & Athanasopoulos, G. (2021)
    Forecasting: principles and practice, 3rd edition, OTexts: Melbourne, Australia.
    OTexts.com/fpp3/useful-predictors.html#fourier-series. Accessed on 05-22-2024.

    Examples
    --------
    Start by loading a timeseries dataset.

    >>> from statsmodels.datasets import co2
    >>> data = co2.load().data

    You can resample the data to whatever frequency you want.

    >>> data = data.resample("ME").mean().ffill()

    Use FFTPeriodicityDetector to find the list of periods using FFT, ordered
    by corresponding frequency amplitudes in a descending order.

    >>> fft_detector = FFTPeriodicityDetector(data)
    >>> periods = fft_detector.fit()

    You can optionally specify a window function for pre-processing.

    >>> periods = fft_detector.fit(window_func="blackman")
    """

    def __init__(self, endog: ArrayLike):
        self.y = to_1d_array(endog)

    def fit(
        self,
        max_period_count: Optional[int] = None,
        detrend_func: Optional[Union[str, Callable[[ArrayLike], NDArray]]] = "linear",
        window_func: Optional[Union[float, str, tuple]] = None,
    ) -> NDArray:
        """
        Find periods in the given series.

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

        See Also
        --------
        numpy.fft
            Discrete Fourier Transform.
        scipy.signal.detrend
            Remove linear trend along axis from data.
        scipy.signal.get_window
            Return a window of a given length and type.

        Returns
        -------
        NDArray
            List of detected periods.
        """
        # Detrend data
        self.y = self.y if detrend_func is None else detrend(self.y, detrend_func)

        # Apply the window function on the data
        self.y = (
            self.y
            if window_func is None
            else apply_window(self.y, window_func=window_func)
        )

        # Compute DFT and ignore the zero frequency
        freqs = np.fft.rfftfreq(len(self.y), d=1)[1:]
        ft = np.fft.rfft(self.y)[1:]

        # Compute periods and their respective amplitudes
        periods = np.round(1 / freqs)
        amps = abs(ft)

        # A period cannot be greater than half the length of the series
        filter = periods < len(self.y) // 2

        # Return periods in descending order of their corresponding amplitudes
        return periods[filter][np.argsort(-amps[filter])][:max_period_count]
