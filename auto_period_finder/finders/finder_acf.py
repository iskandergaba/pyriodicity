from typing import Dict, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pandas import DataFrame, Series
from scipy.signal import argrelmax
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import acf

from auto_period_finder.enums import TimeSeriesDecomposer
from auto_period_finder.tools import (
    remove_overloaded_kwargs,
    seasonality_strength,
    to_1d_array,
)


class AutocorrelationPeriodFinder:
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

    Use AutocorrelationPeriodFinder to find the list of seasonality periods based on ACF.

    >>> period_finder = AutocorrelationPeriodFinder(data)
    >>> periods = period_finder.fit()

    You can also find the most prominent period either ACF-wise or variance-wise.

    >>> strongest_period_acf = period_finder.fit_find_strongest_acf()
    >>> strongest_period_var = period_finder.fit_find_strongest_var()
    """

    def __init__(
        self,
        endog: Union[ArrayLike, DataFrame, Series],
        acf_kwargs: Optional[Dict[str, Union[int, bool, None]]] = None,
    ):
        self.y = to_1d_array(endog)
        self._acf_kwargs = self.__remove_overloaded_acf_kwargs(
            acf_kwargs if acf_kwargs else {}
        )

    def fit(
        self,
        max_period_count: Optional[Union[int, None]] = None,
    ) -> NDArray:
        """
        Find seasonality periods of the given time series automatically.

        Parameters
        ----------
        max_period_count : int, optional, default = None
            Maximum number of periods to look for.

        Returns
        -------
        NDArray
            List of periods.
        """
        return self.__find_periods(self.y, max_period_count, self._acf_kwargs)

    def fit_find_strongest_acf(self) -> np.int64:
        """
        Find the strongest seasonality period ACF-wise of the given time series.

        Returns
        -------
        np.int64
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
        np.int64
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
            p: seasonality_strength(d.seasonal, d.resid) for p, d in decomps.items()
        }
        return max(strengths, key=strengths.get)

    def __find_periods(
        self,
        y: Union[ArrayLike, DataFrame, Series],
        max_period_count: Optional[Union[int, None]],
        acf_kwargs: Dict[str, Union[int, bool, None]],
    ) -> NDArray:
        # Calculate the autocorrelation function
        acf_arr = acf(y, nlags=len(y) // 2, **acf_kwargs)

        # Find local maxima of the first half of the ACF array
        local_argmax = argrelmax(acf_arr)[0]

        # Arg. sort the local maxima in the ACF array in a descending order
        periods = local_argmax[acf_arr[local_argmax].argsort()][::-1]

        # Return the requested maximum count of detectde periods
        return periods[:max_period_count]

    @staticmethod
    def __remove_overloaded_acf_kwargs(acf_kwargs: Dict) -> Dict:
        args = ["x", "nlags", "qstat", "alpha", "bartlett_confint"]
        return remove_overloaded_kwargs(acf_kwargs, args)

    @staticmethod
    def __remove_overloaded_stl_kwargs(stl_kwargs: Dict) -> Dict:
        args = ["endog", "period"]
        return remove_overloaded_kwargs(stl_kwargs, args)

    @staticmethod
    def __remove_overloaded_seasonal_decompose_kwargs(
        seasonal_decompose_kwargs: Dict,
    ) -> Dict:
        args = ["x", "period"]
        return remove_overloaded_kwargs(seasonal_decompose_kwargs, args)
