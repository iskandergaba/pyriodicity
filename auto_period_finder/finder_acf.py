from enum import Enum
from typing import Dict, Optional, Union

import numpy as np
from scipy.signal import argrelmax
from statsmodels.tools.typing import ArrayLike1D
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import acf

from .enums import TimeSeriesDecomposer


class AcfPeriodFinder:
    """
    Autocorrelation function (ACF) based seasonality periods automatic finder.

    Find the periods of a given time series using its ACF. A time delta
    is considered a period if it is a local maximum of ACF.\n

    Parameters
    ----------
    endog : array_like
        Data to be investigated. Must be squeezable to 1-d.
    acf_kwargs: dict, optional
        Arguments to pass to the ACF.

    See Also
    --------
    statsmodels.tsa.stattools.acf
        Autocorrelation function.
    statsmodels.tsa.seasonal.STL
        Season-Trend decomposition using LOESS.
    statsmodels.tsa.seasonal.seasonal_decompose
        Seasonal decomposition using moving averages.

    References
    ----------
    .. [1] Hyndman, R.J., & Athanasopoulos, G. (2021)
    Forecasting: principles and practice, 3rd edition, OTexts: Melbourne, Australia.
    OTexts.com/fpp3/stlfeatures.html. Accessed on 12-23-2023.

    Examples
    --------
    Start by loading a timeseries dataset with a frequency.

    >>> from statsmodels.datasets import co2
    >>> data = co2.load().data

    You can resample the data to whatever frequency you want.

    >>> data = data.resample("M").mean().ffill()

    Use AutoPeriodFinder to find the list of seasonality periods based on ACF.

    >>> period_finder = AutoPeriodFinder(data)
    >>> periods = period_finder.fit()

    You can also find the most prominent period either ACF-wise or variance-wise.

    >>> strongest_period_acf = period_finder.fit_find_strongest_acf()
    >>> strongest_period_var = period_finder.fit_find_strongest_var()
    """

    def __init__(
        self,
        endog: ArrayLike1D,
        acf_kwargs: Optional[Dict[str, Union[int, bool, None]]] = None,
    ):
        self.y = self.__to_1d_array(endog)
        self._acf_kwargs = self.__remove_overloaded_acf_kwargs(
            acf_kwargs if acf_kwargs else {}
        )

    def fit(
        self,
        max_period_count: Optional[Union[int, None]] = None,
    ) -> np.ndarray:
        """
        Find seasonality periods of the given time series automatically.

        Parameters
        ----------
        max_period_count : int, optional, default = None
            Maximum number of periods to look for.

        Returns
        -------
        list
            List of periods.
        """
        return self.__find_periods(self.y, max_period_count, self._acf_kwargs)

    def fit_find_strongest_acf(self) -> np.int64:
        """
        Find the strongest seasonality period ACF-wise of the given time series.

        Parameters
        ----------
        vicinity_radius : int, optional, default = None
            How many data points, before and after, a period candidate
            value to consider for satisfying the periodicity conditions.
            Essentially, the algorithm will verify that at least one point
            in the vicinity (defined by this parameter) of every multiple
            of the candidate value is a local maximum.
            This helps mitigate the effects of the forward and backward
            noise shifts of the period value. It is also effective
            at reducing the number of detected period values that are
            too tightly bunched together.

        Returns
        -------
        int
            The strongest seasonality period ACF-wise.
        """
        periods = self.fit(max_period_count=1)
        if len(periods) == 0:
            return None
        else:
            return periods[0]

    def fit_find_strongest_var(
        self,
        max_period_count: Optional[Union[int, None]] = None,
        decomposer: Optional[
            TimeSeriesDecomposer
        ] = TimeSeriesDecomposer.MOVING_AVERAGES,
        decomposer_kwargs: Optional[Dict[str, Union[int, bool, None]]] = None,
    ) -> np.int64:
        """
        Find the strongest seasonality period variance-wise of the given time series
        using seasonal decomposition.

        Parameters
        ----------
        vicinity_radius : int, optional, default = None
            How many data points, before and after, a period candidate
            value to consider for satisfying the periodicity conditions.
            Essentially, the algorithm will verify that at least one point
            in the vicinity (defined by this parameter) of every multiple
            of the candidate value is a local maximum.
            This helps mitigate the effects of the forward and backward
            noise shifts of the period value. It is also effective
            at reducing the number of detected period values that are
            too tightly bunched together.
        max_period_count : int, optional, default = None
            Maximum number of periods to look for.
        decomposer: TimeSeriesDecomposer. optional, default = TimeSeriesDecomposer.MOVING_AVERAGE_DECOMPOSER
            The seasonality decomposer that returns DecomposeResult to be used to
            determine the strongest seasonality period. The possible values are
            [TimeSeriesDecomposer.MOVING_AVERAGE_DECOMPOSER, TimeSeriesDecomposer.STL].
        decomposer_kwargs: dict, optional
            Arguments to pass to the decomposer.

        Returns
        -------
        int
            The strongest seasonality period.
        """
        periods = self.fit(max_period_count=max_period_count)
        if len(periods) == 0:
            return None
        elif len(periods) == 1:
            return periods[0]
        else:
            if decomposer == TimeSeriesDecomposer.STL:
                decomposer_kwargs = self.__remove_overloaded_stl_kwargs(
                    decomposer_kwargs if decomposer_kwargs else {}
                )
                decomps = {
                    p: STL(self.y, p, **decomposer_kwargs).fit() for p in periods
                }
            elif decomposer == TimeSeriesDecomposer.MOVING_AVERAGES:
                decomposer_kwargs = self.__remove_overloaded_seasonal_decompose_kwargs(
                    decomposer_kwargs if decomposer_kwargs else {}
                )
                decomps = {
                    p: seasonal_decompose(self.y, period=p, **decomposer_kwargs)
                    for p in periods
                }
            else:
                raise ValueError("Invalid seasonality decomposer: " + decomposer)
        strengths = {
            p: self.__seasonality_strength(d.seasonal, d.resid)
            for p, d in decomps.items()
        }
        return max(strengths, key=strengths.get)

    def __find_periods(
        self,
        y: ArrayLike1D,
        max_period_count: Optional[Union[int, None]],
        acf_kwargs: Dict[str, Union[int, bool, None]],
    ) -> list:
        periods = []
        # max_period_count = -1 if max_period_count is None else max_period_count
        acf_arr = np.array(acf(y, nlags=len(y), **acf_kwargs))

        # Find local maxima of the first half of the ACF array
        periods = argrelmax(acf_arr[: y.size // 2])[0]

        return periods[:max_period_count]

    @staticmethod
    def __seasonality_strength(seasonal, resid):
        return max(0, 1 - np.var(resid) / np.var(seasonal + resid))

    @staticmethod
    def __remove_overloaded_acf_kwargs(acf_kwargs: Dict) -> Dict:
        args = ["x", "nlags", "qstat", "alpha", "bartlett_confint"]
        for arg in args:
            acf_kwargs.pop(arg, None)
        return acf_kwargs

    @staticmethod
    def __remove_overloaded_stl_kwargs(stl_kwargs: Dict) -> Dict:
        args = ["endog", "period"]
        for arg in args:
            stl_kwargs.pop(arg, None)
        return stl_kwargs

    @staticmethod
    def __remove_overloaded_seasonal_decompose_kwargs(
        seasonal_decompose_kwargs: Dict,
    ) -> Dict:
        args = ["x", "period"]
        for arg in args:
            seasonal_decompose_kwargs.pop(arg, None)
        return seasonal_decompose_kwargs

    @staticmethod
    def __to_1d_array(x):
        y = np.ascontiguousarray(np.squeeze(np.asarray(x)), dtype=np.double)
        if y.ndim != 1:
            raise ValueError("y must be a 1d array")
        return y
