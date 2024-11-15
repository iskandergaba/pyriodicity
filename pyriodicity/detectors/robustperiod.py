import datetime
from enum import Enum, unique
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

    @staticmethod
    def detect(
        x: ArrayLike,
        lamb: Union[float, str, LambdaSelection] = LambdaSelection.RAVN_UHLIG,
        c: float = 1.5,
    ) -> NDArray:
        """
        Find periods in the given series.

        Parameters
        ----------
        x : array_like
            Data to be investigated. Must be squeezable to 1-d.
        lamb : float, str, RobustPeriod.LambdaSelection,
            default = RobustPeriod.LambdaSelection.RAVN_UHLIG
            The Hodrick-Prescott filter smoothing parameter. Possible values are either
            a float, a `RobustPeriod.LambdaSelection` value, or one of the following
            string values: ['hodrick-prescott', 'ravn-uhlig']. These represent the
            automatic lambda parameter selection methods by Hodrick and Prescott [1]_
            and Ravn and Uhlig [2]_, respectively. If lamb is not float value, then
            ``x`` must be a data array with a datetime-like index.
        c : float, default = 1.5
            The constant threshold that determines the robustness of the Huber function.
            A smaller value makes the Huber function more sensitive to outliers. Huber
            recommends using a value between 1 and 2 [3]_.

        Returns
        -------
        NDArray
            List of detected periods.

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
        """

        # Preprocess the data
        y = RobustPeriod._preprocess(
            x, RobustPeriod.LambdaSelection(lamb) if isinstance(lamb, str) else lamb, c
        )

        # TODO Decouple multiple periodicities

        # TODO Robust single periodicity detection

    @staticmethod
    def _preprocess(
        x: ArrayLike, lamb: Union[float, LambdaSelection], c: float
    ) -> NDArray:
        """
        Apply the data preprocessing step of RobustPeriod.

        Parameters
        ----------
        x : ArrayLike
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
            lamb = RobustPeriod._compute_lambda(x, lamb)

        # Convert to one-dimensional array
        y = to_1d_array(x)

        # Apply Hodrick-Prescott filter
        y, _ = RobustPeriod._hpfilter(y, lamb)

        # Remove outliers using Huber function
        mean = np.mean(y)
        mad = np.mean(np.abs(y - mean))
        return RobustPeriod._huber((y - mean) / mad, c)

    @staticmethod
    def _compute_lambda(x: ArrayLike, lambda_selection: LambdaSelection) -> float:
        """
        Compute the lambda parameter for Hodrick-Prescott filter application on a time
        series.

        Parameters
        ----------
        x : ArrayLike
            Input data array with a datetime-like index.
        lambda_selection : LambdaSelection
            The selection method for the lambda parameter. Must be one of the
            `RobustPeriod.LambdaSelection` enum values.

        Returns
        -------
        float
            Computed lambda parameter value.

        Raises
        ------
        AttributeError
            If the input data does not have an 'index' attribute.
        TypeError
            If the index values are not of type 'numpy.datetime64' or 'datetime.date'.
        ValueError
            If an invalid lambda parameter value is provided.
        """
        if not hasattr(x, "index"):
            raise AttributeError("Data has no attribute 'index'.")
        if not isinstance(x.index[0], (np.datetime64, datetime.date)):
            raise TypeError(
                "Index values are not of 'numpy.datetime64'" "or 'datetime.date' types."
            )
        yearly_nobs = np.rint(np.timedelta64(365, "D") / np.diff(x.index.values).mean())
        if lambda_selection == RobustPeriod.LambdaSelection.HODRICK_PRESCOTT:
            lambda_selection = 100 * yearly_nobs**2

        elif lambda_selection == RobustPeriod.LambdaSelection.RAVN_UHLIG:
            lambda_selection = 6.25 * yearly_nobs**4
        else:
            raise ValueError(
                "Invalid lamb parameter value: '{}'".format(lambda_selection)
            )

    @staticmethod
    def _hpfilter(x: ArrayLike, lamb: float):
        """
        Apply the Hodrick-Prescott filter to a series.

        Parameters
        ----------
        x : ArrayLike
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
        x : ArrayLike
            Input array-like object containing numerical values.
        c : float
            The constant threshold that determines the robustness of the Huber function.
            A smaller value makes the Huber function more sensitive to outliers. Huber
            recommends using a value between 1 and 2.

        Returns
        -------
        ArrayLike
            An array-like object with the Huber function applied element-wise.
        """
        return np.sign(x) * np.min(np.abs(x), c)
