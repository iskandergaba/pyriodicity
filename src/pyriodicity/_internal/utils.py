from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import get_window, periodogram


@staticmethod
def to_1d_array(x: ArrayLike) -> NDArray:
    """
    Convert input to a contiguous 1-dimensional numpy array.

    Parameters
    ----------
    x : array_like
        Input array to be converted. Must be squeezable to 1-d.

    Returns
    -------
    NDArray
        A contiguous 1-dimensional numpy array of type double.

    Raises
    ------
    ValueError
        If the input cannot be squeezed to 1-d.
    """

    x = np.ascontiguousarray(np.squeeze(np.asarray(x)), dtype=np.double)
    if x.ndim != 1:
        raise ValueError("x must be a 1-dimensional array")
    return x


@staticmethod
def apply_window(x: ArrayLike, window_func: str | float | tuple) -> NDArray:
    """
    Apply a window function to the input array.

    Parameters
    ----------
    x : array_like
        Input array. Must be squeezable to 1-d.
    window_func : float, str, tuple
        Window function to apply. See ``scipy.signal.get_window`` for accepted formats
        of the ``window`` parameter.

    Returns
    -------
    NDArray
        Input array with the window function applied.

    See Also
    --------
    scipy.signal.get_window
        Get a window function.
    """

    x = to_1d_array(x)
    return x * get_window(window=window_func, Nx=len(x))


@staticmethod
def acf(x: ArrayLike) -> NDArray:
    """
    Compute the autocorrelation function of a signal.

    Uses FFT to compute the autocorrelation efficiently.

    Parameters
    ----------
    x : array_like
        Input array. Must be squeezable to 1-d.

    Returns
    -------
    NDArray
        The normalized autocorrelation function of the input.
        Length is equal to the input length.
    """

    x = to_1d_array(x)
    n = len(x)
    fft = np.fft.fft(x, n=n * 2)
    psd = fft * np.conjugate(fft)
    acf_arr = np.real(np.fft.ifft(psd))
    return acf_arr[:n] / acf_arr[0]


@staticmethod
def power_threshold(
    x: ArrayLike,
    k: int,
    p: int,
    window_func: str | tuple | ArrayLike = "boxcar",
    detrend_func: Literal["constant", "linear"] | None = "linear",
) -> np.floating:
    """
    Compute the power threshold as the p-th percentile of the maximum
    power values of the periodogram of k permutations of the data.

    Parameters
    ----------
    x : array_like
        Data to be investigated. Must be squeezable to 1-d.
    k : int
        The number of times the data is randomly permuted to compute
        the maximum power values.
    p : int
        The percentile value used to compute the power threshold.
        It determines the cutoff point in the sorted list of the maximum
        power values from the periodograms of the permuted data.
        Value must be between 0 and 100 inclusive.
    window_func : float, str, tuple, optional, default = 'boxcar'
        Window function to apply. See ``scipy.signal.get_window`` for accepted formats
        of the ``window`` parameter.
    detrend_func : {'constant', 'linear'}, optional, default = 'linear'
        The kind of detrending to apply. If None, no detrending is applied.

    Returns
    -------
    numpy.floating
        Power threshold of the target data.

    See Also
    --------
    scipy.signal.periodogram
        Estimate power spectral density using a periodogram.
    """

    max_powers = []
    while len(max_powers) < k:
        _, pxx = periodogram(
            np.random.permutation(x), window=window_func, detrend=detrend_func
        )
        max_powers.append(pxx.max())
    max_powers.sort()
    return np.percentile(max_powers, p)
