import datetime
from enum import Enum, unique
from typing import Union

import numpy as np
import pywt
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import dia_matrix, eye
from scipy.sparse.linalg import spsolve

from pyriodicity.tools import to_1d_array


class RobustPeriod:
    """
    RobustPeriod periodicity detector.

    Find the periods in a given signal or series using RobustPeriod [1]_.

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
    """

    @unique
    class LambdaSelection(Enum):
        """
        Enum for selecting the Hodrick-Prescott filter lambda parameter calculation
        method.

        Attributes
        ----------
        HODRICK_PRESCOTT : str
            Use the Hodrick and Prescott method for lambda calculation [1]_.
        RAVN_UHLIG : str
            Use the Ravn and Uhlig method for lambda calculation [2]_.

        References
        ----------
        .. [1] Hodrick, R. J., & Prescott, E. C. (1997).
           Postwar US business cycles: an empirical investigation.
           Journal of Money, Credit, and Banking, 1-16.
           https://doi.org/10.2307/2953682
        .. [2] Ravn, M. O., & Uhlig, H. (2002).
           On adjusting the Hodrick-Prescott filter for the frequency of observations.
           Review of Economics and Statistics, 84(2), 371-376.
           https://doi.org/10.1162/003465302317411604
        """

        HODRICK_PRESCOTT = "hodrick-prescott"
        RAVN_UHLIG = "ravn-uhlig"

        def compute(self, yearly_nobs: int) -> float:
            """
            Compute the lambda value based on the selected method and yearly number of
            observations.

            Parameters
            ----------
            yearly_nobs : int
                The number of observations per year.

            Returns
            -------
            float
                The computed lambda value.

            Raises
            ------
            ValueError
                If the lambda selection method is unknown.
            """
            if self == RobustPeriod.LambdaSelection.HODRICK_PRESCOTT:
                return 100 * yearly_nobs**2
            elif self == RobustPeriod.LambdaSelection.RAVN_UHLIG:
                return 6.25 * yearly_nobs**4
            else:
                raise ValueError("Unknown lambda selection method")

    @staticmethod
    def detect(
        x: ArrayLike,
        lamb: Union[float, str] = "ravn-uhlig",
        c: float = 1.5,
        db_n: int = 10,
    ) -> NDArray:
        """
        Find periods in the given series.

        Parameters
        ----------
        x : array_like
            Data to be investigated. Must be squeezable to 1-d.
        lamb : float, str, default = 'ravn-uhlig'
            The Hodrick-Prescott filter smoothing parameter. Possible values are either
            a `float`, or one of the following `str` values:
            ['hodrick-prescott', 'ravn-uhlig']. These represent the automatic lambda
            parameter selection methods by Hodrick and Prescott [1]_ and Ravn and Uhlig
            [2]_, respectively. If lamb is not float value, then ``x`` must be a data
            array with a datetime-like index.
        c : float, default = 1.5
            The constant threshold that determines the robustness of the Huber function.
            A smaller value makes the Huber function more sensitive to outliers. Huber
            recommends using a value between 1 and 2 [3]_.
        db_n : int, default = 10
            The number of vanishing moments for the Daubechies wavelet [4]_ used to
            compute the Maximal Overlap Discrete Wavelet Transform (MODWT) [5]_. Must
            be an integer between 1 and 38, inclusive.

        Returns
        -------
        NDArray
            List of detected periods.

        Raises
        ------
        AssertionError
            If `db_n` is not between 1 and 38, inclusive.

        References
        ----------
        .. [1] Hodrick, R. J., & Prescott, E. C. (1997).
           Postwar US business cycles: an empirical investigation.
           Journal of Money, credit, and Banking, 1-16.
           https://doi.org/10.2307/2953682
        .. [2] Ravn, M. O., & Uhlig, H. (2002).
           On adjusting the Hodrick-Prescott filter for the frequency of observations.
           Review of economics and statistics, 84(2), 371-376.
           https://doi.org/10.1162/003465302317411604
        .. [3] Huber, P. J., & Ronchetti, E. (2009). Robust Statistics. Wiley.
           https://doi.org/10.1002/9780470434697
        .. [4] Daubechies, I. (1992). Ten lectures on wavelets. Society for industrial
           and applied mathematics.
           https://doi.org/10.1137/1.9781611970104
        .. [5] Percival, D. B. (2000). Wavelet methods for time series analysis.
           Cambridge University Press.
           https://doi.org/10.1017/CBO9780511841040
        """

        # Validate the db_n parameter
        assert 1 <= db_n <= 38, "Invalid db_n parameter value: '{}'".format(db_n)

        # Preprocess the data
        lamb = RobustPeriod.LambdaSelection(lamb) if isinstance(lamb, str) else lamb
        y = RobustPeriod._preprocess(x, lamb, c)

        # TODO Decouple multiple periodicities
        RobustPeriod._decouple_periodicities(y)

        # TODO Robust single periodicity detection

    @staticmethod
    def _preprocess(
        x: ArrayLike, lamb: Union[float, LambdaSelection], c: float
    ) -> NDArray:
        """
        Apply the data preprocessing step of RobustPeriod.

        Parameters
        ----------
        x : array_like
            Data to be preprocessed. Must be squeezable to 1-d.
        lamb : float, LambdaSelection
            The Hodrick-Prescott filter smoothing parameter. If a
            `RobustPeriod.LambdaSelection` value is provided, then ``x`` must be a data
            array with a datetime-like index.
        c : float
            The constant threshold that determines the robustness of the Huber function.
            A smaller value makes the Huber function more sensitive to outliers. Huber
            recommends using a value between 1 and 2.

        Returns
        -------
        NDArray
            Preprocessed data.
        """

        # Compute the lambda parameter if a lambda selection method is provided
        if isinstance(lamb, RobustPeriod.LambdaSelection):
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
            lamb = lamb.compute(yearly_nobs)

        # Convert to one-dimensional array
        y = to_1d_array(x)

        # Apply Hodrick-Prescott filter
        y, _ = RobustPeriod._hpfilter(y, lamb)

        # Remove outliers using Huber function
        mean = np.mean(y)
        mad = np.mean(np.abs(y - mean))
        return RobustPeriod._huber((y - mean) / mad, c)

    @staticmethod
    def _decouple_periodicities(x: ArrayLike, db_n: int, level: int):
        w_coeffs = RobustPeriod._modwt(x, db_n, level)
        bivar = np.array(
            [
                # Exclude the first Lj - 1 coefficients
                RobustPeriod._biweight_midvariance(w_coeffs[j][1][j:], 9)
                for j in range(level)
            ]
        )
        pass

    @staticmethod
    def _hpfilter(x: ArrayLike, lamb: float):
        """
        Apply the Hodrick-Prescott filter to a series.

        Parameters
        ----------
        x : array_like
            The series data to be filtered.
        lamb : float
            The smoothing parameter for the Hodrick-Prescott filter.

        Returns
        -------
        cycle : NDArray
            The cyclical component of the time series.
        trend : NDArray
            The trend component of the time series.
        """

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
        """
        Compute the Huber function for an array-like input.

        Parameters
        ----------
        x : array_like
            Input array-like object containing numerical values.
        c : float
            The constant threshold that determines the robustness of the Huber function.
            A smaller value makes the Huber function more sensitive to outliers. Huber
            recommends using a value between 1 and 2.

        Returns
        -------
        NDArray
            An array-like object with the Huber function applied element-wise.
        """
        return np.sign(x) * np.minimum(np.abs(x), c)

    @staticmethod
    def _modwt(x: ArrayLike, db_n: int, level: int) -> NDArray:
        """
        Compute the Maximal Overlap Discrete Wavelet Transform (MODWT) of a series using
        the Daubechies wavelet.

        Parameters
        ----------
        x : array_like
            Input data to be transformed. Must be squeezable to 1-d.
        db_n : int
            The number of vanishing moments for the Daubechies wavelet. Must be an
            integer between 1 and 38, inclusive.
        level : int
            The number of decomposition steps to perform.

        Returns
        -------
        NDArray
            The MODWT coefficients of the input data.
        """

        # Pad the input data to the nearest 2^level multiple
        padding = (2**level - (len(x) % 2**level)) % 2**level
        y = np.pad(x, (0, padding), "wrap")

        # Compute the Maximal Overlap Discrete Wavelet Transform
        return pywt.swt(y, "db{}".format(db_n), level, norm=True)

    @staticmethod
    def _biweight_midvariance(x: ArrayLike, c: float) -> float:
        # Ensure the correct data type
        x = np.asanyarray(x).astype(np.float64)

        # Compute u
        med = np.median(x)
        mad = np.mean(np.abs(x - np.mean(x)))
        u = (x - med) / (c * mad)

        # Indicator function
        indicator = np.abs(u) < 1

        # Return the biweight midvariance estimation result
        return (
            len(x)
            * np.sum((x[indicator] - med) ** 2 * (1 - u[indicator] ** 2) ** 4)
            / np.sum((1 - u[indicator] ** 2) * (1 - 5 * u[indicator] ** 2)) ** 2
        )
