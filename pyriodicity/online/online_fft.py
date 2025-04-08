from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyriodicity.tools import OnlineHelper


class OnlineFFTPeriodicityDetector:
    """
    Online Fast Fourier Transform (FFT) based periodicity detector.

    Detect periodicities in a signal stream data using Sliding Discrete Fourier
    Transform (DFT) algorithm [1]_.

    Parameters
    ----------
    window_size : int
        Size of the sliding window for the ACF computation.
    window_func : float, str, tuple, optional, default = 'boxcar'
        Window function to apply. See ``scipy.signal.get_window`` for accepted formats
        of the ``window`` parameter.
    detrend_func : {'constant', 'linear'}, optional, default = 'linear'
        The kind of detrending to apply. If None, no detrending is applied.

    See Also
    --------
    pyriodicity.FFTPeriodicityDetector
        Fast Fourier Transform (FFT) based periodicity detector.

    References
    ----------
    .. [1] Hyndman, R.J., & Athanasopoulos, G. (2021)
       Forecasting: principles and practice, 3rd edition, OTexts: Melbourne, Australia.
       https://OTexts.com/fpp3/useful-predictors.html#fourier-series.
       Accessed on 09-15-2024.
    """

    def __init__(
        self,
        window_size: int,
        window_func: Union[float, str, tuple] = "boxcar",
        detrend_func: Optional[Literal["constant", "linear"]] = "linear",
    ):
        # Initialize the online helper
        self.online_helper = OnlineHelper(window_size, window_func, detrend_func)

        # Compute the DFT sample frequencies and exclude the DC frequency
        self.freqs = np.fft.rfftfreq(window_size)[1:]

        # Compute the possible periodicity lengths
        self.periods = np.rint(1 / self.freqs).astype(int)
        self.period_filter = self.periods < window_size // 2 + 1
        self.periods = self.periods[self.period_filter]

    def detect(
        self,
        data: Union[np.floating, ArrayLike],
        max_period_count: Optional[int] = None,
    ) -> NDArray:
        """
        Update the frequency spectrum and detect periodicities.

        Process new samples through the detector's circular buffer, updating the
        frequency spectrum and detecting periodicities in the signal using the SDFT
        algorithm.

        Parameters
        ----------
        data : numpy.floating or array_like
            New samples to process. Can be a single value or an array of values.
            Multi-dimensional arrays will be flattened.
        max_period_count : int, optional, default = None
            Maximum number of periods to return. If None, all detected periods are
            returned.

        Returns
        -------
        numpy.ndarray
            Array of detected periods sorted by their amplitude strength in descending
            order.
        """

        # Update the frequency spectrum
        spectrum = self.online_helper.update(data, return_value="rfft")

        # Compute the frequency amplitudes
        amps = abs(spectrum[1:])
        amps = amps[self.period_filter]

        # Sort period length values in the descending order of their amplitudes
        result = self.periods[np.argsort(-amps)]

        # Return unique period length values
        _, unique_indices = np.unique(result, return_index=True)
        return result[np.sort(unique_indices)][:max_period_count]
