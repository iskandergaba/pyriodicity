from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import find_peaks

from pyriodicity.tools import OnlineHelper


class OnlineACFPeriodicityDetector:
    """
    Find periods in streaming signal data using Sliding DFT algorithm.

    Parameters
    ----------
    window_size : int
        Size of the sliding window (should be a power of 2 for best performance).
    window_func : float or str or tuple
        Window function to apply. Default is None (rectangular window). See
        ``scipy.signal.get_window`` for accepted formats of the ``window`` parameter.
    detrend_func : {'constant', 'linear'}, optional
        The kind of detrending to apply. Default is 'linear'. If None,
        no detrending is applied.
    max_period_count : int, optional
        Maximum number of periods to return. Default is None (return all periods).

    Notes
    -----
    Uses Sliding DFT for efficient online computation of frequency spectrum
    and period detection in streaming data.
    """

    def __init__(
        self,
        window_size: int,
        window_func: Union[float, str, tuple] = "boxcar",
        detrend_func: Optional[Literal["constant", "linear"]] = "linear",
        max_period_count: Optional[int] = None,
    ):
        # Store the detector variables
        self.window_size = window_size
        self.max_period_count = max_period_count

        # Initialize the online helper
        self.online_helper = OnlineHelper(window_size, window_func, detrend_func)

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

        # Compute the ACF
        acf_arr = self.online_helper.update(data, return_value="acf")

        # Find peaks in the first half of the ACF array, excluding the first element
        peaks, properties = find_peaks(acf_arr[1 : self.window_size // 2], height=-1)
        peak_heights = properties["peak_heights"]

        # Sort peaks by height in descending order and account for the excluded element
        periods = peaks[np.argsort(peak_heights)[::-1]] + 1

        # Return the requested maximum count of detected periods
        return periods[: self.max_period_count]
