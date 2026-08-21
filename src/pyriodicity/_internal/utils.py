from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import get_window, periodogram


def to_1d_array(x: ArrayLike) -> NDArray[np.floating]:
    """
    Convert input to a contiguous 1-dimensional numpy array.

    Parameters
    ----------
    x : array_like
        Input array to be converted. Must be squeezable to 1-d.

    Returns
    -------
    ndarray
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


def apply_window(
    x: NDArray[np.floating], window_func: str | float | tuple
) -> NDArray[np.floating]:
    """
    Apply a window function to the input array.

    Parameters
    ----------
    x : array_like
        Input array. Must be a 1-d array.
    window_func : str, float, tuple
        Window function to apply. See ``scipy.signal.get_window`` for accepted formats
        of the ``window`` parameter.

    Returns
    -------
    ndarray
        Input array with the window function applied.

    See Also
    --------
    scipy.signal.get_window
        Get a window function.
    """

    return x * get_window(window=window_func, Nx=len(x))


def acf(x: ArrayLike) -> NDArray[np.floating]:
    """
    Compute the autocorrelation function of a signal.

    Uses FFT to compute the autocorrelation efficiently.

    Parameters
    ----------
    x : array_like
        Input array. Must be squeezable to 1-d.

    Returns
    -------
    ndarray
        The normalized autocorrelation function of the input.
        Length is equal to the input length.
    """

    x = to_1d_array(x)
    n = len(x)
    fft = np.fft.fft(x, n=n * 2)
    psd = fft * np.conjugate(fft)
    acf_arr = np.real(np.fft.ifft(psd))
    return acf_arr[:n] / acf_arr[0]


def power_threshold(
    x: ArrayLike,
    k: int,
    p: int,
    window_func: str | tuple | ArrayLike = "boxcar",
    detrend_func: Literal["constant", "linear"] | None = "linear",
    seed: int | None = None,
) -> float:
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
    window_func : str, tuple, array_like, optional, default = 'boxcar'
        Window function to apply. See ``scipy.signal.periodogram`` for accepted formats
        of the ``window`` parameter.
    detrend_func : {'constant', 'linear'}, optional, default = 'linear'
        The kind of detrending to apply. If None, no detrending is applied.
    seed : int, optional, default = None
        A seed or generator to make the random permutations reproducible. See
        ``numpy.random.default_rng`` for the accepted values.

    Returns
    -------
    float
        Power threshold of the target data.

    See Also
    --------
    scipy.signal.periodogram
        Estimate power spectral density using a periodogram.
    """

    max_powers = []
    rng = np.random.default_rng(seed)
    while len(max_powers) < k:
        _, pxx = periodogram(
            rng.permutation(x),
            window=window_func,
            detrend=detrend_func,
        )
        max_powers.append(pxx.max())
    max_powers.sort()
    return np.percentile(max_powers, p)
