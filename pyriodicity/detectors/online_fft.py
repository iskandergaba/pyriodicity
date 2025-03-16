from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import detrend

from pyriodicity.tools import apply_window, to_1d_array


class OnlineFFTPeriodicityDetector:
    """
    TODO
    """

    def __init__(
        self,
        window_size: int,
        max_period_count: Optional[int] = None,
        detrend_func: Optional[Literal["constant", "linear"]] = "linear",
        window_func: Optional[Union[float, str, tuple]] = None,
    ):
        # Initialize
        self.max_period_count = max_period_count
        self.detrend_func = detrend_func
        self.N = window_size

        # Initialize the window
        self.window = (
            np.ones(self.N)
            if window_func is None
            else apply_window(np.ones(self.N), window_func)
        )

        # Compute the twiddle factors
        self.twiddle = np.exp(-2j * np.pi * np.arange(self.N // 2 + 1) / self.N)

        # Initialize the buffer for time domain samples (real-valued)
        self.buffer = np.zeros(self.N)
        self.buffer_idx = 0

        # Compute the initial spectrum and exclude the DC component
        self.spectrum = np.fft.rfft(self.buffer)

        # Compute the DFT sample frequencies and exclude the DC frequency
        self.freqs = np.fft.rfftfreq(self.N)[1:]

        # Compute the possible periodicity lengths
        self.periods = np.round(1 / self.freqs).astype(int)
        self.period_filter = self.periods < self.N // 2 + 1
        self.periods = self.periods[self.period_filter]

    def detect(self, data: Union[np.floating, ArrayLike]) -> NDArray:
        """Push a new value to the buffer.
        TODO
        """
        data = to_1d_array(data)
        for sample in np.array(data).flat:
            if self.detrend_func is not None:
                # TODO detrend if needed
                pass

            # Swap the oldest for the newest sample at the current buffer index
            old_sample = self.buffer[self.buffer_idx]
            self.buffer[self.buffer_idx] = sample

            # Apply the window function on the newest and oldest samples
            sample *= self.window[self.buffer_idx]
            old_sample *= self.window[self.buffer_idx]

            # Update the spectrum
            self.spectrum = self.twiddle * (self.spectrum + sample - old_sample)

            # Update the buffer index
            self.buffer_idx = (self.buffer_idx + 1) % self.N

        # Compute the frequency amplitudes
        amps = abs(self.spectrum[1:])
        amps = amps[self.period_filter]

        # Sort period length values in the descending order of their amplitudes
        result = self.periods[np.argsort(-amps)]

        # Return unique period length values
        _, unique_indices = np.unique(result, return_index=True)
        return result[np.sort(unique_indices)][: self.max_period_count]
