from typing import Callable, Dict, List, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pandas import DataFrame, Series
from scipy.signal import detrend as _detrend
from scipy.signal import get_window
from scipy.stats import kendalltau, pearsonr, spearmanr


@staticmethod
def to_1d_array(x: Union[ArrayLike, DataFrame, Series]) -> NDArray:
    y = np.ascontiguousarray(np.squeeze(np.asarray(x)), dtype=np.double)
    if y.ndim != 1:
        raise ValueError("y must be a 1d array")
    return y


@staticmethod
def remove_overloaded_kwargs(kwargs: Dict, args: List) -> Dict:
    for arg in args:
        kwargs.pop(arg, None)
    return kwargs


# Deprecated
@staticmethod
def apply_window_fun(x: ArrayLike, window_func: Union[str, float, tuple]) -> NDArray:
    return (x - np.median(x)) * get_window(window=window_func, Nx=len(x))


@staticmethod
def seasonality_strength(seasonal: ArrayLike, resid: ArrayLike) -> float:
    return max(0, 1 - np.var(resid) / np.var(seasonal + resid))


@staticmethod
def apply_window(x: ArrayLike, window_func: Union[str, float, tuple]) -> NDArray:
    return x * get_window(window=window_func, Nx=len(x))


@staticmethod
def detrend(
    x: Union[ArrayLike, DataFrame, Series],
    method: Union[str, Callable[[Union[ArrayLike, DataFrame, Series]], NDArray]],
) -> NDArray:
    if isinstance(method, str):
        return _detrend(x, type=method)
    return method(x)


@staticmethod
def acf(
    x: Union[ArrayLike, DataFrame, Series],
    nlags: int,
    correlation_func: Optional[str] = "pearson",
) -> NDArray:
    if not 0 < nlags <= len(x):
        raise ValueError("nlags must be a postive integer less than the data length")
    if correlation_func == "spearman":
        return np.array([spearmanr(x, np.roll(x, l)).statistic for l in range(nlags)])
    elif correlation_func == "kendall":
        return np.array([kendalltau(x, np.roll(x, l)).statistic for l in range(nlags)])
    return np.array([pearsonr(x, np.roll(x, l)).statistic for l in range(nlags)])
