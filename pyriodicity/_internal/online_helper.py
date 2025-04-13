from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import detrend, get_window, ShortTimeFFT


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
    buffer_size : int, optional, default = 2 * window_size
        Size of the samples buffer. Must be at least equal to window_size.
    window_func : float or str or tuple, optional, default = 'boxcar'
        Window function to apply. See ``scipy.signal.get_window`` for accepted formats
        of the ``window`` parameter.
    detrend_func : {'constant', 'linear'}, optional, default = 'linear'
        The kind of detrending to apply. If None, no detrending is applied.

    Attributes
    ----------
    window : NDArray
        The window function array.
    window_buffer : NDArray
        Circular buffer storing the last ``window_size`` windowed signal samples.
    buffer : NDArray
        Circular buffer storing the last ``buffer_size`` samples.
    rfft : NDArray
        Current FFT spectrum of the windowed signal.

    See Also
    --------
    scipy.signal.get_window
        Get a window function with the specified parameters.

    Raises
    ------
    ValueError
        If history_size is less than window_size.
    """

    def __init__(
        self,
        window_size: int,
        buffer_size: Optional[int] = None,
        window_func: Union[str, float, tuple] = "boxcar",
        detrend_func: Optional[Literal["constant", "linear"]] = "linear",
    ):
        # Initialize size attributes
        self.window_size = window_size
        self.buffer_size = 2 * window_size if buffer_size is None else buffer_size
        if self.buffer_size < window_size:
            raise ValueError(
                f"History size ({self.buffer_size}) must be at least "
                f"equal to window size ({window_size})"
            )

        # Initialize the buffers
        self.window_buffer = np.zeros(window_size)
        self.buffer = np.zeros(self.buffer_size)
        self._buffer_index = 0

        # Initialize the detrend function attribute
        self._detrend_func = detrend_func

        # Get the window
        self.window = get_window(window=window_func, Nx=window_size)

        # Compute the twiddle factors
        self._twiddle = np.exp(
            -2j * np.pi * np.arange(window_size // 2 + 1) / window_size
        )

        # Initialize the FFT spectrum and frequencies
        self.rfft = np.fft.rfft(self.window_buffer)
        self.freq = np.fft.rfftfreq(self.window_size)

        # Initialize Short Time FFT
        self._stft = ShortTimeFFT.from_window(
            win_param=window_func, fs=1.0, nperseg=self.window_size, noverlap=0
        )

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

        data_array = np.asarray(data).flat
        n_samples = len(data_array)

        for i in range(0, n_samples, self.window_size):
            batch = data_array[i : i + self.window_size]
            batch_size = len(batch)

            # Update history buffer
            self.buffer = np.roll(self.buffer, -batch_size)
            self.buffer[-batch_size:] = batch

            # Get old samples and update window buffer
            old_samples = self.window_buffer[:batch_size].copy()
            self.window_buffer = np.roll(self.window_buffer, -batch_size)
            self.window_buffer[-batch_size:] = batch

            # Detrend if needed
            if self._detrend_func is not None:
                detrended_buffer = detrend(
                    np.insert(self.window_buffer, 0, old_samples),
                    type=self._detrend_func,
                )
                old_samples = detrended_buffer[:batch_size]
                batch = detrended_buffer[-batch_size:]

            # Apply window
            old_samples *= self.window[:batch_size]
            self.window = np.roll(self.window, -batch_size)
            batch *= self.window[-batch_size:]

            # Update spectrum using twiddle factors for the batch
            shift_indices = np.arange(batch_size)[:, None] * np.arange(
                self.window_size // 2 + 1
            )
            shifts = np.exp(-2j * np.pi * shift_indices / self.window_size)
            delta_spectrum = (batch - old_samples) @ shifts
            self.rfft = self._twiddle**batch_size * (self.rfft + delta_spectrum)

            # Update buffer index
            self._buffer_index = (self._buffer_index + batch_size) % self.buffer_size

            # Recompute spectrum when buffer is filled
            if self._buffer_index == 0:
                stft_buffer = (
                    self.buffer
                    if self._detrend_func is None
                    else detrend(self.buffer, type=self._detrend_func)
                )
                S = self._stft.stft(stft_buffer)
                self.rfft = S[:, -1]

        match return_value:
            case "rfft":
                return self.rfft
            case "acf":
                return self.get_acf()
            case _:
                raise ValueError(f"Unsupported return_value '{return_value}'")

    def get_acf(self) -> NDArray:
        """
        Get the current autocorrelation function array.

        Computes the ACF from the current FFT spectrum using the inverse FFT
        and normalizes it by the zero-lag autocorrelation.

        Returns
        -------
        NDArray
            The normalized autocorrelation function array of the current window buffer.
        """

        acf_arr = np.fft.irfft(self.rfft)
        return np.zeros_like(acf_arr) if acf_arr[0] == 0 else acf_arr / acf_arr[0]
