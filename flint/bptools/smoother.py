import numpy as np
from scipy.signal import savgol_filter

from flint.logging import logger


def smooth_data(
    data: np.ndarray, window_size: int, polynomial_order: int
) -> np.ndarray:
    """Smooth a 1-dimensional dataset. Internally it uses a savgol filter as
    implemented in scipy.signal.savgol_filter. It is intended to be used to
    smooth the real and imaginary components of the complex gains of the
    bandpass solutions.

    Datapoints that are NaN's are first filled by linearly interpolation the
    closest valid data points. Once the savgol filter has been applied these
    datapoints are then remasked with a NaN.

    Args:
        data (np.ndarray): The 1-dimensional data to be smoothed.
        window_size (int): The size of the window function of the savgol filter. Passed directly to savgol.
        polynomial_order (int): The order of the polynomial of the savgol filter. Passed directly to savgol.

    Returns:
        np.ndarray: Smoothed dataset
    """

    # Make a copy so we do not mess around with the original numpy data
    # where ever it might be. Trust nothing you sea dog.
    data = data.copy()

    # Before we smooth we need to fill in channels that are flagged with nans.
    # For this we will apply a simply linear interpolation across the blanked
    # regions, smooth, and then reflag them later
    x = np.arange(len(data))
    mask = ~np.isfinite(data)
    d_interp = np.interp(x[mask], x[~mask], data[~mask])

    # This is the filling in of the blanks
    data[mask] = d_interp

    # Now we smooth. This savgol filter fits a polynomial to succesive subsets of data
    # in a manner similar to a box car. The final positional argument here denoted the
    # behavour of the edge where the window (second positional argument) does not have
    # enough data. This process is similar to the original implemented in bptools, except
    # here we are using a polynomial, not a set of harmonic basis functions.
    smoothed_data = savgol_filter(data, window_size, polynomial_order)
    smoothed_data[mask] = np.nan

    return smoothed_data


def smooth_bandpass_complex_gains(
    complex_gains: np.ndarray, window_size: int = 16, polynomial_order: int = 1
) -> np.ndarray:
    """Smooth bandpass solutions by applying a savgol filter to the real and imaginary components
    of each of the antenna based polarisation solutions across channels.

    Experience suggests that this performs better once the data have been divided by a reference
    antenna. Flagged channels (represented by NaNs) will be maintained in the smoothed
    output data.

    The input bandpass data contained by ``complex_gains`` is expected to be in the form:
    > [ants, chans, pols]

    Args:
        complex_gains (np.ndarray): Data to be smoothed.
        window_size (int, optional): The size of the window function of the savgol filter. Passed directly to savgol. Defaults to 16.
        polynomial_order (int, optional): The order of the polynomial of the savgol filter. Passed directly to savgol. Defaults to 1.

    Returns:
        np.ndarray: Smoothed complex gains
    """

    assert (
        len(complex_gains.shape) == 3
    ), f"The shape of the input complex gains should be of rank 3 in form (ant, chan, pol). Received {complex_gains.shape}"

    smoothed_complex_gains = np.zeros_like(complex_gains) * np.nan

    ants, chans, pols = complex_gains.shape

    logger.info(f"Smoothing using {window_size=} {polynomial_order=}")
    logger.critical(
        "BE CAREFUL - there are no constraints to prevent smoothing over spectral window boundaries. "
    )
    logger.critical(
        "Consider whether you really want to be smoothing when using continuum data. "
    )
    for ant in range(ants):
        for pol in range(pols):
            smoothed_complex_gains[ant, :, pol].real = smooth_data(
                data=complex_gains[ant, :, pol].real,
                window_size=window_size,
                polynomial_order=polynomial_order,
            )
            smoothed_complex_gains[ant, :, pol].imag = smooth_data(
                data=complex_gains[ant, :, pol].imag,
                window_size=window_size,
                polynomial_order=polynomial_order,
            )

    return smoothed_complex_gains
