import datetime
from concurrent.futures import ProcessPoolExecutor
from enum import Enum, unique
from functools import partial
from multiprocessing import cpu_count
from typing import Literal, Optional, Tuple, Union

import numpy as np
import pywt
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.signal import find_peaks
from scipy.sparse import dia_matrix, eye
from scipy.sparse.linalg import spsolve
from scipy.special import binom, huber

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
       (pp. 2328-2337). https://doi.org/10.1145/3448016.3452779

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

    class _HuberPeriodogram:
        @classmethod
        def compute(
            cls,
            x: NDArray,
            delta: float,
            max_worker_count: int,
        ) -> NDArray:
            """
            Compute the Huber M-Periodogram using ADMM with parallel execution.

            Parameters
            ----------
            x : array_like
                Input data to be transformed. Must be squeezable to 1-d.
            delta : float
                The tuning constant for the Huber loss function.

            Returns
            -------
            NDArray
                The Huber M-Periodogram of the input data.
            """

            with ProcessPoolExecutor(max_worker_count) as executor:
                periodogram = list(
                    executor.map(
                        partial(cls._compute_element, x, delta=delta), range(len(x))
                    )
                )

            return np.array(periodogram)

        @classmethod
        def _compute_element(cls, x: NDArray, k: int, delta: float) -> np.floating:
            n = len(x)
            t = np.arange(n)
            phi = np.array(
                [np.cos(2 * np.pi * k * t / n), np.sin(2 * np.pi * k * t / n)]
            ).T

            # Huber Robust M-Periodogram objective function
            def objective(beta):
                return np.linalg.norm(huber(delta, phi @ beta - x.T))

            result = minimize(objective, np.zeros(phi.shape[1]))
            return n * np.linalg.norm(result.x) ** 2 / 4

    @unique
    class _LambdaSelection(Enum):
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
            if self == RobustPeriod._LambdaSelection.HODRICK_PRESCOTT:
                return 100 * yearly_nobs**2
            elif self == RobustPeriod._LambdaSelection.RAVN_UHLIG:
                return 6.25 * yearly_nobs**4
            else:
                raise ValueError("Unknown lambda selection method")

    @staticmethod
    def detect(
        data: ArrayLike,
        lamb: Union[float, Literal["hodrick-prescott", "ravn-uhlig"]] = "ravn-uhlig",
        c: float = 1.5,
        db_n: int = 8,
        modwt_level: int = 10,
        delta: float = 1.345,
        max_worker_count: Optional[int] = None,
        max_period_count: Optional[int] = None,
    ) -> NDArray:
        """
        Find periods in the given series.

        Parameters
        ----------
        data : array_like
            Data to be investigated. Must be squeezable to 1-d.
        lamb : float, {'hodrick-prescott', 'ravn-uhlig'}, default = 'ravn-uhlig'
            The Hodrick-Prescott filter smoothing parameter. Possible values are either
            a `float`, 'hodrick-prescott', or 'ravn-uhlig'. These represent the lambda
            parameter selection heuristics by Hodrick and Prescott [1]_ and Ravn and
            Uhlig [2]_, respectively. If lamb is not float value, then ``x`` must be a
            data array with a datetime-like index.
        c : float, default = 1.5
            The constant threshold that determines the robustness of the Huber function.
            A smaller value makes the Huber function more sensitive to outliers. Huber
            recommends using a value between 1 and 2 [3]_.
        db_n : int, default = 8
            The number of vanishing moments for the Daubechies wavelet [4]_ used to
            compute the Maximal Overlap Discrete Wavelet Transform (MODWT) [5]_. Must
            be an integer between 1 and 38, inclusive.
        modwt_level : int, default = 10
            The level of the Maximal Overlap Discrete Wavelet Transform (MODWT). Must be
            a positive integer.
        delta : float, default = 1.345
            The tuning constant for the Huber loss function. A smaller value makes the
            function more sensitive to outliers [3]_.
        max_worker_count : int, optional
            The maximum number of worker processes to use. Defaults to the number of
            CPUs in the system.
        max_period_count : int, optional
            The maximum number of periods to detect. If not specified, all detected
            periods are returned.

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

        # Validate db_n parameter
        assert 1 <= db_n <= 38, "Invalid db_n parameter value: '{}'".format(db_n)

        # Validate max_worker_count parameter
        max_worker_count = cpu_count() if max_worker_count is None else max_worker_count

        # Validate max_period_count parameter
        max_period_count = modwt_level if max_period_count is None else max_period_count

        x = RobustPeriod._preprocess(
            data,
            RobustPeriod._LambdaSelection(lamb) if isinstance(lamb, str) else lamb,
            c,
        )

        # Decouple multiple periodicities
        w_coeff_list = RobustPeriod._wavelet_coeffs(x, db_n, modwt_level)

        # Robust single periodicity detection
        return RobustPeriod._detect(
            w_coeff_list, delta, max_worker_count, max_period_count
        )

    @staticmethod
    def _preprocess(
        x: ArrayLike, lamb: Union[float, _LambdaSelection], c: float
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

        def hpfilter(x: NDArray, lamb: float) -> Tuple:
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

            nobs = len(x)

            # Identity matrix
            identity = eye(nobs, nobs)

            # Second-order difference matrix
            offsets = np.array([0, 1, 2])
            data = np.repeat([[1.0], [-2.0], [1.0]], nobs, axis=1)
            K = dia_matrix((data, offsets), shape=(nobs - 2, nobs))

            # Solve the linear system
            trend = spsolve(identity + lamb * K.T.dot(K), x)
            cycle = x - trend
            return cycle, trend

        def huber_function(x: ArrayLike, c: float) -> NDArray:
            """
            Compute the Huber function for an array-like input.

            Parameters
            ----------
            x : array_like
                Data to be preprocessed. Must be squeezable to 1-d.
            c : float
                The constant threshold that determines the robustness of the Huber
                function. A smaller value makes the Huber function more sensitive to
                outliers. Huber recommends using a value between 1 and 2.

            Returns
            -------
            NDArray
                An array-like object with the Huber function applied element-wise.
            """
            return np.sign(x) * np.minimum(np.abs(x), c)

        # Compute the lambda parameter if a lambda selection method is provided
        if isinstance(lamb, RobustPeriod._LambdaSelection):
            index = getattr(x, "index", None)
            if index is None or not hasattr(index, "values"):
                raise AttributeError("Data has no attribute 'index'.")

            index_values = index.values
            if not isinstance(index_values[0], (np.datetime64, datetime.date)):
                raise TypeError(
                    "Index values are not of 'numpy.datetime64'"
                    "or 'datetime.date' types."
                )
            yearly_nobs = np.rint(
                np.timedelta64(365, "D") / np.diff(index_values).mean()
            )
            lamb = lamb.compute(yearly_nobs)

        # Convert to one-dimensional array
        y = to_1d_array(x)

        # Apply Hodrick-Prescott filter
        y, _ = hpfilter(y, lamb)

        # Remove outliers using Huber function
        mean = np.mean(y)
        mad = np.mean(np.abs(y - mean))
        return huber_function((y - mean) / mad, c)

    @staticmethod
    def _wavelet_coeffs(x: NDArray, db_n: int, level: int):
        """
        Compute the wavelet coefficients for a given series using the Maximal Overlap
        Discrete Wavelet Transform (MODWT) and the Daubechies wavelet.

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
            The wavelet coefficients ordered in the descending order of their variances.
        """

        def modwt(x: NDArray, db_n: int, level: int) -> NDArray:
            """
            Compute the Maximal Overlap Discrete Wavelet Transform (MODWT) of a series
            using the Daubechies wavelet.

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
            coeffs = pywt.swt(y, "db{}".format(db_n), level, norm=True)
            return np.array([cD[: len(x)] for _, cD in coeffs])

        def biweight_midvariance(x: NDArray, c: float) -> float:
            """
            Compute the biweight midvariance of a given array.

            Parameters
            ----------
            x : array-like
                The input array for which the biweight midvariance is to be computed.
            c : float
                The tuning constant that determines the robustness and efficiency of
                the estimator.

            Returns
            -------
            float
                The biweight midvariance of the input array.
            """

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

        # Compute wavelet coefficients using MODWT
        w_coeffs = modwt(x, db_n, level)

        # Compute wavelet variances
        w_vars = np.array(
            [
                # Exclude the first Lj - 1 coefficients
                biweight_midvariance(w_coeffs[j][j:], 9)
                for j in range(level)
            ]
        )

        # Order wavelet coefficients in the descending order of their variances
        return w_coeffs[np.argsort(-w_vars)]

    @staticmethod
    def _detect(
        w_coeff_list: NDArray,
        delta: float,
        max_worker_count: int,
        max_period_count: int,
    ) -> NDArray:
        """
        Detect periods in the given wavelet coefficient list.

        Parameters
        ----------
        w_coeff_list : array_like
            List of wavelet coefficients to be analyzed.
        delta : float
            The tuning constant for the Huber loss function. A smaller value makes the
            function more sensitive to outliers.
        max_worker_count : int
            The maximum number of worker processes to use for parallel processing.
        max_period_count : int
            The maximum number of periods to detect. If None, all detected periods are
            returned.

        Returns
        -------
        NDArray
            List of detected periods.
        """

        def fisher_g_test(g0: float, n: int) -> float:
            """
            Perform Fisher's exact test for a given g-statistic and sample size.

            Parameters
            ----------
            g0 : float
                The g-statistic value. Must be between 0 and 1 (exclusive).
            n : int
                The sample size.

            Returns
            -------
            float
                The p-value resulting from Fisher's exact test.

            Raises
            ------
            AssertionError
                If g0 is not within the range (0, 1].
            """
            # Validate the g0 parameter
            assert 0 < g0 <= 1, "Invalid g0 parameter value: '{}'".format(g0)

            k = np.arange(1, int(1 // g0) + 1)
            return np.sum(
                np.nan_to_num(binom(n, k)) * np.nan_to_num((1 - k * g0) ** (n - 1))
            )

        def get_period(periodogram: NDArray) -> Optional[int]:
            """
            Determine the period of a given periodogram.

            Parameters
            ----------
            periodogram : array-like
                The input periodogram for which the period is to be determined.

            Returns
            -------
            int
                The detected period length of the input periodogram. Returns None if
                the period cannot be determined.

            Notes
            -----
            This function computes the modified autocorrelation function (ACF) using the
            Huber loss function and identifies the period by finding peaks in the
            rescaled ACF.
            """

            def huber_acf(periodogram: NDArray) -> NDArray:
                """
                Compute the modified autocorrelation function (ACF) for a given
                periodogram using the Huber loss function.

                Parameters
                ----------
                periodogram : array-like
                    The input periodogram for which the ACF is to be computed.

                Returns
                -------
                NDArray
                    The modified autocorrelation function of the input periodogram.
                """

                n_prime = len(periodogram)
                n = n_prime // 2

                # Compute P_bar
                part_1 = periodogram[:n]
                part_2 = (
                    np.sum(
                        periodogram[range(0, n_prime, 2)]
                        - periodogram[range(1, n_prime, 2)]
                    )
                    ** 2
                    / n_prime
                )
                part_3 = np.flip(part_1[1:])
                p_bar = np.hstack([part_1, part_2, part_3])

                # Compute P
                p = np.real(np.fft.ifft(p_bar))

                return p[:n] / ((n - np.arange(0, n)) * p[0])

            # Compute and rescale the Huber ACF
            n_prime = len(periodogram)
            acf = huber_acf(periodogram)
            acf_rescaled = (acf - acf.min()) / (acf.max() - acf.min())

            # The periodicity must be less than n // 2, i.e. less than n_prime // 4
            k = np.argmax(periodogram)
            if k < 4:
                return None

            # Compute the period
            peaks, _ = find_peaks(acf_rescaled, height=0.5)
            distances = np.diff(peaks)
            period = np.median(distances).astype(int) if len(distances) > 0 else 0
            r_k = (
                0.5 * ((n_prime / (k + 1)) + (n_prime / k)) - 1,
                0.5 * ((n_prime / k) + (n_prime / (k - 1))) + 1,
            )
            return period if r_k[0] <= period <= r_k[1] else None

        # Compute the periodograms
        periodograms = [
            RobustPeriod._HuberPeriodogram.compute(
                np.pad(w_coeffs, (0, len(w_coeffs))), delta, max_worker_count
            )
            for w_coeffs in w_coeff_list
        ]

        # Filter out periodograms not containing periodic components
        periodograms = [
            pg
            for pg in periodograms
            if fisher_g_test(np.max(pg) / np.sum(pg), len(pg)) < 0.05
        ]

        # Compute the periods
        periods = []
        for pg in periodograms:
            if max_period_count <= len(periods):
                break
            p = get_period(pg)
            if p is not None and p not in periods:
                periods.append(p)
        return np.array(periods)
