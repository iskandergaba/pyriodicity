from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pandas import DataFrame, Series
from scipy.signal import get_window


@staticmethod
def to_1d_array(x: Union[ArrayLike, DataFrame, Series]) -> NDArray:
    y = np.ascontiguousarray(np.squeeze(np.asarray(x)), dtype=np.double)
    if y.ndim != 1:
        raise ValueError("y must be a 1d array")
    return y


@staticmethod
def apply_window_fun(x: ArrayLike, window_func: Union[str, float, tuple]) -> NDArray:
    return (x - np.median(x)) * get_window(window=window_func, Nx=len(x))


@staticmethod
def seasonality_strength(seasonal: ArrayLike, resid: ArrayLike) -> float:
    return max(0, 1 - np.var(resid) / np.var(seasonal + resid))
