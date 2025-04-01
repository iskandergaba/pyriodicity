from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import find_peaks

from pyriodicity.tools import OnlineHelper


class OnlineACFPeriodicityDetector:
    """
    Online Autocorrelation Function (ACF) based periodicity detector.

    Parameters
    ----------
    window_size : int
        Size of the sliding window for the ACF computation.
    window_func : float, str, tuple, optional, default = 'boxcar'
        Window function to apply. See ``scipy.signal.get_window`` for accepted formats
        of the ``window`` parameter.
    detrend_func : {'constant', 'linear'}, optional, default = 'linear'
        The kind of detrending to apply. If None, no detrending is applied.

    Notes
    -----
    This detector uses an online approach to compute the ACF, allowing for efficient
    periodicity detection in streaming data.
    """

    def __init__(
        self,
        window_size: int,
        window_func: Union[float, str, tuple] = "boxcar",
        detrend_func: Optional[Literal["constant", "linear"]] = "linear",
    ):
        self.window_size = window_size
        self.online_helper = OnlineHelper(window_size, window_func, detrend_func)

    def detect(
        self,
        data: Union[np.floating, ArrayLike],
        max_period_count: Optional[int] = None,
    ) -> NDArray:
        """
        Detect periods in a signal using online ACF.

        Process new samples through the detector's circular buffer, updating the
        ACF and detecting periodic patterns in the signal.

        Parameters
        ----------
        data : numpy.floating or array_like
            New samples to process. Can be a single value or an array of values.
        max_period_count : int, optional, default = None
            Maximum number of periods to return. If None, all detected periods are
            returned.

        Returns
        -------
        NDArray
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
        * Computes the ACF
        * Finds peaks in the ACF to identify periods

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
        return periods[: max_period_count]
