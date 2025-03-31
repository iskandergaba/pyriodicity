from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pyriodicity.tools import OnlineHelper


class OnlineFFTPeriodicityDetector:
    """
    Find periods in streaming signal data using Sliding DFT algorithm.

    Parameters
    ----------
    window_size : int
        Size of the sliding window (should be a power of 2 for best performance).
    max_period_count : int, optional
        Maximum number of periods to return. Default is None (return all periods).
    detrend_func : {'constant', 'linear'}, optional
        The kind of detrending to apply. Default is 'linear'. If None,
        no detrending is applied.
    window_func : float or str or tuple, optional
        Window function to apply. Default is None (rectangular window). See
        ``scipy.signal.get_window`` for accepted formats of the ``window`` parameter.

    Notes
    -----
    Uses Sliding DFT for efficient online computation of frequency spectrum
    and period detection in streaming data.
    """

    def __init__(
        self,
        window_size: int,
        max_period_count: Optional[int] = None,
        detrend_func: Optional[Literal["constant", "linear"]] = "linear",
        window_func: Optional[Union[float, str, tuple]] = None,
    ):
        # Store the detector variables
        self.max_period_count = max_period_count

        # Initialize the online helper
        self.online_helper = OnlineHelper(window_size, detrend_func, window_func)

        # Compute the DFT sample frequencies and exclude the DC frequency
        self.freqs = np.fft.rfftfreq(window_size)[1:]

        # Compute the possible periodicity lengths
        self.periods = np.rint(1 / self.freqs).astype(int)
        self.period_filter = self.periods < window_size // 2 + 1
        self.periods = self.periods[self.period_filter]

    def detect(self, data: Union[np.floating, ArrayLike]) -> NDArray:
        """
        Detect periods in a signal using Sliding DFT with online updates.

        Process new samples through the detector's circular buffer, updating the
        frequency spectrum and detecting periodic patterns in the signal using
        the Sliding DFT algorithm.

        Parameters
        ----------
        data : numpy.floating or array_like
            New samples to process. Can be a single value or an array of values.
            Multi-dimensional arrays will be flattened.

        Returns
        -------
        numpy.ndarray
            Array of detected periods sorted by their amplitude strength in
            descending order. Only unique periods are returned, limited by
            max_period_count if specified. Each period represents the length
            (in samples) of a detected periodicity.

        Notes
        -----
        The detection process follows these steps:

        * Updates the circular buffer
        * Applies detrending if specified
        * Applies windowing if specified
        * Updates the frequency spectrum
        * Computes periods from the spectrum

        Only periods shorter than ``window_size // 2 + 1`` are considered reliable
        and returned.
        """

        # Update the frequency spectrum
        spectrum = self.online_helper.rfft(data)

        # Compute the frequency amplitudes
        amps = abs(spectrum[1:])
        amps = amps[self.period_filter]

        # Sort period length values in the descending order of their amplitudes
        result = self.periods[np.argsort(-amps)]

        # Return unique period length values
        _, unique_indices = np.unique(result, return_index=True)
        return result[np.sort(unique_indices)][: self.max_period_count]
