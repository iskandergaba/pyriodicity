import datetime
from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import dia_matrix, eye
from scipy.sparse.linalg import spsolve

from pyriodicity.tools import to_1d_array


class RobustPeriod:
    """
    RobustPeriod periodicity detector.

    Find the periods in a given signal or series using RobustPeriod [1]_.

    See Also
    --------
    pyriodicity.Autoperiod
        Autoperiod periodicity detector.

    References
    ----------
    .. [1] Wen, Q., He, K., Sun, L., Zhang, Y., Ke, M., & Xu, H. (2021, June).
       RobustPeriod: Robust time-frequency mining for multiple periodicity detection.
       In Proceedings of the 2021 international conference on management of data
       (pp. 2328-2337). https://doi.org/10.48550/arXiv.2002.09535

    Examples
    --------
    Start by loading Mauna Loa Weekly Atmospheric CO2 Data from
    `statsmodels <https://statsmodels.org>`_ and downsampling its data to a monthly
    frequency.

    >>> from statsmodels.datasets import co2
    >>> data = co2.load().data
    >>> data = data.resample("ME").mean().ffill()

    Use ``RobustPeriod`` to find the list of periods in the data.

    >>> from pyriodicity import RobustPeriod
    >>> RobustPeriod.detect(data)
    array([12])

    You can specify a lower percentile value should you wish for
    a more lenient detection

    >>> RobustPeriod.detect(data, percentile=90)
    array([12])

    You can also increase the number of random data permutations
    for a more robust power threshold estimation

    >>> RobustPeriod.detect(data, k=300)
    array([12])

    ``RobustPeriod`` is considered a more robust variant of ``Autoperiod``
    against noise. The detection algorithm found exactly one periodicity
    length of 12, suggesting a strong yearly periodicity.
    """

    @staticmethod
    def detect(
        endog: ArrayLike,
        lamb: Union[str, float] = "ravn-uhlig",
        detrend_func: Optional[str] = "linear",
        window_func: Optional[Union[str, float, tuple]] = None,
        correlation_func: Optional[str] = "pearson",
    ) -> NDArray:
        """
        Find periods in the given series.

        Parameters
        ----------
        endog : array_like
            Data to be investigated. Must be squeezable to 1-d.
        k : int, optional, default = 100
            The number of times the data is randomly permuted while estimating the
            power threshold.
        percentile : int, optional, default = 99
            Percentage for the percentile parameter used in computing the power
            threshold. Value must be between 0 and 100 inclusive.
        detrend_func : str, default = 'linear'
            The kind of detrending to be applied on the signal. It can either be
            'linear' or 'constant'.
        window_func : float, str, tuple, optional, default = None
            Window function to be applied to the time series. Check
            ``window`` parameter documentation for ``scipy.signal.get_window``
            function for more information on the accepted formats of this
            parameter.
        correlation_func : str, default = 'pearson'
            The correlation function to be used to calculate the ACF of the time
            series. Possible values are ['pearson', 'spearman', 'kendall'].

        Returns
        -------
        NDArray
            List of detected periods.

        See Also
        --------
        scipy.signal.detrend
            Remove linear trend along axis from data.
        scipy.signal.get_window
            Return a window of a given length and type.
        scipy.stats.kendalltau
            Calculate Kendall's tau, a correlation measure for ordinal data.
        scipy.stats.pearsonr
            Pearson correlation coefficient and p-value for testing non-correlation.
        scipy.stats.spearmanr
            Calculate a Spearman correlation coefficient with associated p-value.

        """
        y = to_1d_array(endog)

        # Detrend data
        # y = y if detrend_func is None else detrend(y, type=detrend_func)
        # Apply window on data
        # y = y if window_func is None else apply_window(y, window_func)

        # Preprocess the data
        y = RobustPeriod._preprocess(y, lamb)

        # TODO Decouple multiple periodicities

        # TODO Robust single periodicity detection

    @staticmethod
    def _preprocess(
        x: ArrayLike,
        lamb: Union[str, float],
    ) -> bool:
        """
        Validate the period hint.

        Parameters
        ----------
        y : array_like
            Data to be investigated. Must be squeezable to 1-d.
        hint : float
            The period hint to be validated.
        detrend_func : str
            The kind of detrending to be applied on the signal. It can either be
            'linear' or 'constant'.
        correlation_func : str
            The correlation function to be used to calculate the ACF of the series
            or the signal. Possible values are ['pearson', 'spearman', 'kendall'].

        Returns
        -------
        bool
            Whether the period hint is valid.
        """

        # Apply Hodrick-Prescott filter
        y, _ = RobustPeriod._hpfilter(x, lamb=lamb)

        # Remove outliers using Huber function
        mean = np.mean(y)
        mad = np.mean(np.abs(y - mean))
        return RobustPeriod._huber((y - mean) / mad, 2)

    @staticmethod
    def _hpfilter(x: ArrayLike, lamb: Union[str, float]):
        """
        Apply the Hodrick-Prescott filter to a series.

        Parameters
        ----------
        x : array_like
            The time series to be filtered.
        lamb : float, str, optional
            The smoothing parameter. Default is 1600.

        Returns
        -------
        cycle : ndarray
            The cyclical component of the time series.
        trend : ndarray
            The trend component of the time series.
        """

        # Compute lamb if required
        if isinstance(lamb, str):
            if not hasattr(x, "index"):
                raise AttributeError("Data has no attribute 'index'.")
            if not isinstance(x.index[0], (np.datetime64, datetime.date)):
                raise TypeError(
                    "Index values are not of 'numpy.datetime64'"
                    "or 'datetime.date' types."
                )
            yearly_nobs = np.rint(
                np.timedelta64(365, "D") / np.diff(x.index.values).mean()
            )
            if lamb == "hodrick-prescott":
                lamb = 100 * yearly_nobs**2
            elif lamb == "ravn-uhlig":
                lamb = 6.25 * yearly_nobs**4
            else:
                raise ValueError("Invalid lamb parameter value: '{}'".format(lamb))

        y = np.asarray(x)
        nobs = len(y)

        # Identity matrix
        identity = eye(nobs, nobs)

        # Second-order difference matrix
        offsets = np.array([0, 1, 2])
        data = np.repeat([[1.0], [-2.0], [1.0]], nobs, axis=1)
        K = dia_matrix((data, offsets), shape=(nobs - 2, nobs))

        # Solve the linear system
        trend = spsolve(identity + lamb * K.T.dot(K), y)
        cycle = y - trend
        return cycle, trend

    @staticmethod
    def _huber(x: ArrayLike, c: float) -> ArrayLike:
        # TODO Research the choice of c
        return np.sign(x) * np.min(np.abs(x), c)
