from typing import Literal, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import get_window, periodogram


@staticmethod
def to_1d_array(x: ArrayLike) -> NDArray:
    x = np.ascontiguousarray(np.squeeze(np.asarray(x)), dtype=np.double)
    if x.ndim != 1:
        raise ValueError("x must be a 1-dimensional array")
    return x


@staticmethod
def apply_window(x: ArrayLike, window_func: Union[str, float, tuple]) -> NDArray:
    x = to_1d_array(x)
    return x * get_window(window=window_func, Nx=len(x))


@staticmethod
def acf(x: ArrayLike) -> NDArray:
    x = to_1d_array(x)
    _, power_spectrum = periodogram(x)
    acf = np.fft.irfft(power_spectrum)
    return acf / acf[0]


@staticmethod
def power_threshold(
    x: ArrayLike,
    detrend_func: Literal["constant", "linear"],
    k: int,
    p: int,
) -> np.floating:
    """
    Compute the power threshold as the p-th percentile of the maximum
    power values of the periodogram of k permutations of the data.

    Parameters
    ----------
    x : array_like
        Data to be investigated. Must be squeezable to 1-d.
    detrend_func : {'constant', 'linear'}
        The kind of detrending to be applied on the signal. If None, no detrending
        is applied.
    k : int
        The number of times the data is randomly permuted to compute
        the maximum power values.
    p : int
        The percentile value used to compute the power threshold.
        It determines the cutoff point in the sorted list of the maximum
        power values from the periodograms of the permuted data.
        Value must be between 0 and 100 inclusive.

    See Also
    --------
    scipy.signal.periodogram
        Estimate power spectral density using a periodogram.

    Returns
    -------
    float
        Power threshold of the target data.
    """
    max_powers = []
    while len(max_powers) < k:
        _, power_p = periodogram(np.random.permutation(x), detrend=detrend_func)
        max_powers.append(power_p.max())
    max_powers.sort()
    return np.percentile(max_powers, p)
