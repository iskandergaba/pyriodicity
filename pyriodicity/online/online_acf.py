from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import find_peaks

from pyriodicity.tools import OnlineHelper


class OnlineACFPeriodicityDetector:
    """
    Online Autocorrelation Function (ACF) based periodicity detector.

    Detect periodicities in a signal stream using online ACF. A lag value
    is considered a period if it is a local maximum of the ACF [1]_.

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
    pyriodicity.ACFPeriodicityDetector
        Autocorrelation Function (ACF) based periodicity detector.

    References
    ----------
    .. [1] Hyndman, R.J., & Athanasopoulos, G. (2021)
       Forecasting: principles and practice, 3rd edition, OTexts: Melbourne, Australia.
       https://OTexts.com/fpp3/acf.html. Accessed on 09-15-2024.
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
        Update the online ACF and detect periodicities.

        Process new samples through the detector's circular buffer, updating the ACF
        and detecting periodicities in the signal.

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
            Array of detected periodicity lengths, sorted by strength in descending
            order.
        """

        # Compute the ACF
        acf_arr = self.online_helper.update(data, return_value="acf")

        # Find peaks in the first half of the ACF array, excluding the first element
        peaks, properties = find_peaks(acf_arr[1 : self.window_size // 2], height=-1)
        peak_heights = properties["peak_heights"]

        # Sort peaks by height in descending order and account for the excluded element
        periods = peaks[np.argsort(peak_heights)[::-1]] + 1

        # Return the requested maximum count of detected periods
        return periods[:max_period_count]
