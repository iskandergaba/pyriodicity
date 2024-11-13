import datetime
from typing import Union

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
        c: float = 2,
    ) -> NDArray:
        """
        Find periods in the given series.

        Parameters
        ----------
        endog : array_like
            Data to be investigated. Must be squeezable to 1-d.
        lamb : float, str, default = 'ravn-uhlig'
            TODO explanation
        c : float, default = 2
            TODO explanation

        Returns
        -------
        NDArray
            List of detected periods.

        See Also
        --------
        TODO explanation

        """
        y = to_1d_array(endog)

        # Preprocess the data
        y = RobustPeriod._preprocess(y, lamb, c)

        # TODO Decouple multiple periodicities

        # TODO Robust single periodicity detection

    @staticmethod
    def _preprocess(x: ArrayLike, lamb: Union[str, float], c: float) -> NDArray:
        """
        Validate the period hint.

        Parameters
        ----------
        x : array_like
            Data to be preprocessed. Must be squeezable to 1-d.
        lamb : float, str, default = 'ravn-uhlig'
            TODO explanation
        c : float, default = 2
            TODO explanation

        Returns
        -------
        NDArray
            Preprocessed series data.
        """

        # Apply Hodrick-Prescott filter
        y, _ = RobustPeriod._hpfilter(x, lamb=lamb)

        # Remove outliers using Huber function
        mean = np.mean(y)
        mad = np.mean(np.abs(y - mean))
        return RobustPeriod._huber((y - mean) / mad, c)

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
