from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import detrend, get_window


class OnlineHelper:
    def __init__(
        self,
        window_size: int,
        window_func: Union[str, float, tuple] = "boxcar",
        detrend_func: Optional[Literal["constant", "linear"]] = "linear",
    ):
        self.window_size = window_size
        self.detrend_func = detrend_func

        # Get the window
        self.window = get_window(window=window_func, Nx=window_size)

        # Compute the twiddle factors
        self.twiddle = np.exp(
            -2j * np.pi * np.arange(window_size // 2 + 1) / window_size
        )

        # Initialize the buffer for time domain samples
        self.buffer = np.zeros(window_size)

        # Initialize the spectrum
        self.spectrum = np.fft.rfft(self.buffer)

    def update(
        self,
        data: Union[np.floating, ArrayLike],
        return_value: Literal["rfft", "acf"] = "rfft",
    ) -> NDArray:
        for sample in np.asarray(data).flat:
            # Swap the oldest for the newest sample
            old_sample = self.buffer[0]
            self.buffer[0] = sample
            self.buffer = np.roll(self.buffer, -1)

            # Detrend data
            if self.detrend_func is not None:
                detrended_buffer = detrend(
                    np.insert(self.buffer, 0, old_sample), type=self.detrend_func
                )
                old_sample = detrended_buffer[0]
                sample = detrended_buffer[-1]

            # Apply the window function on the oldest and newest samples
            old_sample *= self.window[0]
            self.window = np.roll(self.window, -1)
            sample *= self.window[0]

        # Update the spectrum
        self.spectrum = self.twiddle * (self.spectrum + sample - old_sample)

        # Return the requested value
        match return_value:
            case "rfft":
                return self.get_rfft()
            case "acf":
                return self.get_acf()
            case _:
                raise ValueError(f"Unsupported return_value '{return_value}'")

    def get_freq(self) -> NDArray:
        return np.fft.rfftfreq(self.window_size)

    def get_rfft(self) -> NDArray:
        return self.spectrum

    def get_acf(self) -> NDArray:
        # Compute the ACF using the inverse FFT
        acf_arr = np.fft.irfft(self.get_rfft())
        # Return the normalized ACF
        return np.zeros_like(acf_arr) if acf_arr[0] == 0 else acf_arr / acf_arr[0]
