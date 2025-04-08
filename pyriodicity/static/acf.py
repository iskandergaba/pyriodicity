from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import detrend, find_peaks

from pyriodicity.tools import acf, apply_window, to_1d_array


class ACFPeriodicityDetector:
    """
    Autocorrelation Function (ACF) based periodicity detector.

    Find the periods in a given signal or series using its ACF. A lag value
    is considered a period if it is a local maximum of the ACF [1]_.

    See Also
    --------
    pyriodicity.OnlineACFPeriodicityDetector
        online Autocorrelation Function (ACF) based periodicity detector.

    References
    ----------
    .. [1] Hyndman, R.J., & Athanasopoulos, G. (2021)
       Forecasting: principles and practice, 3rd edition, OTexts: Melbourne, Australia.
       https://OTexts.com/fpp3/acf.html. Accessed on 09-15-2024.

    Examples
    --------
    Start by loading Mauna Loa Weekly Atmospheric CO2 Data from
    `statsmodels <https://statsmodels.org>`_ and downsampling its data to a monthly
    frequency.

    >>> from statsmodels.datasets import co2
    >>> data = co2.load().data
    >>> data = data.resample("ME").mean().ffill()

    Use ACFPeriodicityDetector to find the list of seasonality periods using the ACF.

    >>> from pyriodicity import ACFPeriodicityDetector
    >>> ACFPeriodicityDetector.detect(data)
    array([ 12,  24,  36,  48,  60,  72,  84,  96, 108, 120, 132, 143, 155,
       167, 179, 191, 203, 215, 227, 239, 251])

    All of the returned values are either multiples of 12 or very close to it,
    suggesting a clear yearly periodicity.
    You can also get the most prominent period length value by setting
    ``max_period_count`` to 1.

    >>> ACFPeriodicityDetector.detect(data, max_period_count=1)
    array([12])
    """

    @staticmethod
    def detect(
        data: ArrayLike,
        window_func: Union[str, float, tuple] = "boxcar",
        detrend_func: Optional[Literal["constant", "linear"]] = "linear",
        max_period_count: Optional[int] = None,
    ) -> NDArray:
        """
        Find periods in the given series.

        Parameters
        ----------
        data : array_like
            Data to be investigated. Must be squeezable to 1-d.
        window_func : float, str, tuple, optional, default = 'boxcar'
            Window function to apply. See ``scipy.signal.get_window`` for accepted
            formats of the ``window`` parameter.
        detrend_func : {'constant', 'linear'}, optional, default = 'linear'
            The kind of detrending to apply. If None, no detrending is applied.
        max_period_count : int, optional, default = None
            Maximum number of periods to return. If None, all detected periods are
            returned.

        Returns
        -------
        NDArray
            Array of detected periodicity lengths, sorted by strength in descending
            order

        See Also
        --------
        scipy.signal.get_window
            Return a window of a given length and type.
        """

        x = to_1d_array(data)

        # Detrend data
        x = x if detrend_func is None else detrend(x, type=detrend_func)

        # Apply window on data
        x = apply_window(x, window_func)

        # Compute the ACF
        acf_arr = acf(x)

        # Find peaks in the first half of the ACF array, excluding the first element
        peaks, properties = find_peaks(acf_arr[1 : len(x) // 2], height=-1)
        peak_heights = properties["peak_heights"]

        # Sort peaks by height in descending order and account for the excluded element
        periods = peaks[np.argsort(peak_heights)[::-1]] + 1

        # Return the requested maximum count of detected periodicity lengths
        return periods[:max_period_count]
