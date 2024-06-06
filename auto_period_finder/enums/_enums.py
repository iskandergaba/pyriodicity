from enum import Enum


class TimeSeriesDecomposer(Enum):
    """
    Time series seasonality decomposition method.
    """

    STL = 1
    MOVING_AVERAGES = 2
