import math
from enum import Enum
from typing import Dict, Optional, Union

import numpy as np
from statsmodels.tools.typing import ArrayLike1D
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import acf


class Decomposer(Enum):
    """
    Seasonality decomposition method.
    """

    STL = 1
    MOVING_AVERAGES = 2


class AutoPeriodFinder:
    """
    Autocorrelation function (ACF) based seasonality periods automatic finder.

    Find the periods of a given time series using its ACF. A time delta
    is considered a period if:
        1- It is a local maximum of ACF.\n
        2- Its multiples are also local maxima.\n
        3- It is not a multiple of an already discovered period. For example,
        it is redundant to return a 2 year seasonality period if we have already
        found a 1 year period. The inverse, however, is not necessarily true.

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
        vicinity_radius: Optional[Union[int, None]] = None,
        max_period_count: Optional[Union[int, None]] = None,
    ) -> list:
        """
        Find seasonality periods of the given time series automatically.

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

        Returns
        -------
        list
            List of periods.
        """
        return self.__find_periods(
            self.y, vicinity_radius, max_period_count, self._acf_kwargs
        )

    def fit_find_strongest_acf(
        self, vicinity_radius: Optional[Union[int, None]] = None
    ) -> int:
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
        periods = self.fit(vicinity_radius=vicinity_radius, max_period_count=1)
        if len(periods) == 0:
            return None
        else:
            return periods[0]

    def fit_find_strongest_var(
        self,
        vicinity_radius: Optional[Union[int, None]] = None,
        max_period_count: Optional[Union[int, None]] = None,
        decomposer: Optional[Decomposer] = Decomposer.MOVING_AVERAGES,
        decomposer_kwargs: Optional[Dict[str, Union[int, bool, None]]] = None,
    ) -> int:
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
        decomposer: Decomposer. optional, default = Decomposer.MOVING_AVERAGE_DECOMPOSER
            The seasonality decomposer that returns DecomposeResult to be used to
            determine the strongest seasonality period. The possible values are
            [Decomposer.MOVING_AVERAGE_DECOMPOSER, Decomposer.STL].
        decomposer_kwargs: dict, optional
            Arguments to pass to the decomposer.

        Returns
        -------
        int
            The strongest seasonality period.
        """
        periods = self.fit(
            vicinity_radius=vicinity_radius, max_period_count=max_period_count
        )
        if len(periods) == 0:
            return None
        elif len(periods) == 1:
            return periods[0]
        else:
            if decomposer == Decomposer.STL:
                decomposer_kwargs = self.__remove_overloaded_stl_kwargs(
                    decomposer_kwargs if decomposer_kwargs else {}
                )
                decomps = {
                    p: STL(self.y, p, **decomposer_kwargs).fit() for p in periods
                }
            elif decomposer == Decomposer.MOVING_AVERAGES:
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
        vicinity_radius: Optional[Union[int, None]],
        max_period_count: Optional[Union[int, None]],
        acf_kwargs: Dict[str, Union[int, bool, None]],
    ) -> list:
        periods = []
        acf_arr = np.array(acf(y, nlags=len(y), **acf_kwargs))
        acf_arr_work = acf_arr.copy()

        # Eliminate the trivial seasonality period of 0 and its vicinity
        vicinity_radius = 0 if vicinity_radius is None else vicinity_radius
        acf_arr_work[0 : vicinity_radius + 1] = -1

        while True:
            # i is a period candidate: It cannot be greater than half the timeseries length
            i = acf_arr_work[: (acf_arr_work.size - vicinity_radius - 1) // 2].argmax()

            # No more periods left or the maximum number of periods has been found
            if acf_arr_work[i] == -1 or (
                max_period_count is not None and len(periods) == max_period_count
            ):
                return periods

            # Check that i and all of its multiples are local maxima
            period = self.__get_period(acf_arr, i, vicinity_radius)
            if period is not None:
                # Add to period return list
                periods.append(period)
                # Ignore i and its multiples
                for offset in np.arange(-vicinity_radius, vicinity_radius + 1):
                    acf_arr_work[
                        [i * j + offset for j in np.arange(1, len(acf_arr_work) // i)]
                    ] = -1

            # Not a period, ignore it
            else:
                acf_arr_work[
                    [
                        i + offset
                        for offset in np.arange(-vicinity_radius, vicinity_radius + 1)
                    ]
                ] = -1

    @staticmethod
    def __get_period(acf_arr, lag, vicinity_radius=0):
        # Lag value vicinity offset range array
        vicinity = np.arange(-vicinity_radius, vicinity_radius + 1)
        # Possible lag value multipliers
        multipliers = np.arange(
            2, math.ceil((len(acf_arr) - vicinity_radius - 1) / lag)
        )
        # The total number of local maxima found
        local_maxima_count = 0
        # Lag value accumulator for local maxima found at the corresponding multiplier indices
        lag_value_acc = np.zeros(len(multipliers), dtype=np.int64)
        for offset in vicinity:
            multiple_is_local_maxima = [
                acf_arr[lag * j + offset - 1] < acf_arr[lag * j + offset]
                and acf_arr[lag * j + offset] > acf_arr[lag * j + offset + 1]
                for j in multipliers
            ]
            local_maxima_count += multiple_is_local_maxima.count(True)
            lag_value_acc[multiple_is_local_maxima] = (
                lag_value_acc[multiple_is_local_maxima] + lag + offset
            )

        # No lag value in the vicinity is a period
        if any(f == 0 for f in lag_value_acc):
            return None
        # Return the average lag value
        else:
            return np.sum(lag_value_acc) // local_maxima_count

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
