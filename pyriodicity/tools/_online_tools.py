from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import detrend, get_window


class OnlineHelper:
    """
    Online helper class for efficient signal processing using sliding windows.

    Implements an efficient sliding window approach for computing real-time FFT
    and ACF of a signal stream. Uses twiddle factors and incremental updates
    to minimize computational complexity.

    Parameters
    ----------
    window_size : int
        Size of the sliding window for signal processing.
    window_func : float or str or tuple, optional, default = 'boxcar'
        Window function to apply. See ``scipy.signal.get_window`` for accepted formats
        of the ``window`` parameter.
    detrend_func : {'constant', 'linear'}, optional, default = 'linear'
        The kind of detrending to apply. If None, no detrending is applied.

    Attributes
    ----------
    window : NDArray
        The window function array.
    buffer : NDArray
        Circular buffer storing the windowed signal samples.
    spectrum : NDArray
        Current FFT spectrum of the windowed signal.
    twiddle : NDArray
        Pre-computed twiddle factors for efficient FFT updates.

    See Also
    --------
    scipy.signal.get_window
        Get a window function with the specified parameters.
    """

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

        # Initialize the FFT spectrum and frequencies
        self.spectrum = np.fft.rfft(self.buffer)
        self.freq = np.fft.rfftfreq(self.window_size)

    def update(
        self,
        data: Union[np.floating, ArrayLike],
        return_value: Literal["rfft", "acf"] = "rfft",
    ) -> NDArray:
        """
        Update the signal buffer with new samples and compute transforms.

        Processes new samples through the circular buffer, updating the FFT spectrum
        efficiently using twiddle factors. Can return either the FFT or ACF of
        the current window.

        Parameters
        ----------
        data : numpy.floating or array_like
            New samples to process. Can be a single value or an array of values.
        return_value : {'rfft', 'acf'}, optional, default = 'rfft'
            The type of transform to return:
            - 'rfft': Return the real FFT spectrum
            - 'acf': Return the autocorrelation function

        Returns
        -------
        NDArray
            The requested transform (FFT spectrum or ACF) of the current window.

        Raises
        ------
        ValueError
            If return_value is not one of {'rfft', 'acf'}.
        """

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

    def get_rfft(self) -> NDArray:
        """
        Get the current FFT spectrum.

        Returns
        -------
        NDArray
            The real FFT spectrum of the current window.
        """

        return self.spectrum

    def get_acf(self) -> NDArray:
        """
        Get the current autocorrelation function.

        Computes the ACF from the current FFT spectrum using the inverse FFT
        and normalizes it by the zero-lag autocorrelation.

        Returns
        -------
        NDArray
            The normalized autocorrelation function of the current window.
        """

        acf_arr = np.fft.irfft(self.get_rfft())
        return np.zeros_like(acf_arr) if acf_arr[0] == 0 else acf_arr / acf_arr[0]
