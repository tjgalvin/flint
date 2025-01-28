from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from flint.logging import logger


def divide_bandpass_by_ref_ant_preserve_phase(
    complex_gains: np.ndarray, ref_ant: int
) -> np.ndarray:
    """Divide the bandpass complex gains (solved for initially by something like
    calibrate) by a nominated reference antenna. In the case of ``calibrate``
    there is no implicit reference antenna. This is valid for cases where the
    xy-phase is set to 0 (true via the ASKAP on-dish calibrator).

    This particular function is most appropriate for the `calibrate` style
    solutions, which solve for the Jones in one step. In HMS notation this
    are normally split into two separate 2x2 matrices, one for the gains
    with zero off-diagonal elements and a leakage matrix with ones on
    the diagonal.

    This is the preferred function to use whena attempting to set a
    phase reference antenna to precomputed Jones bandpass solutions.

    The input complex gains should be in the form:
    >> (ant, channel, pol)

    Internally reference phasores are constructed for the G_x and G_y
    terms of the reference antenna. They are then applied:
    >> G_xp = G_x / G_xref
    >> G_xyp = G_xy / G_yref
    >> G_yxp = G_yx / G_xref
    >> G_y = G_y / G_yref

    which is applied to all antennas in ``complex_gains``.

    Args:
        complex_gains (np.ndarray): The complex gains that will be normalised
        ref_ant (int): The desired reference antenna to use

    Returns:
        np.ndarray: The normalised bandpass solutions
    """
    assert len(complex_gains.shape) == 3, (
        f"The shape of the input complex gains should be of rank 3 in form (ant, chan, pol). Received {complex_gains.shape}"
    )

    logger.info(
        f"Dividing bandpass gain solutions using reference antenna={ref_ant}, using correct phasor"
    )

    # Unpack the values for short hand use
    g_x = complex_gains[:, :, 0]
    g_xy = complex_gains[:, :, 1]
    g_yx = complex_gains[:, :, 2]
    g_y = complex_gains[:, :, 3]

    # In the operations below our ship only wants to be touching
    # the phases in a piratey manner. The amplitudes should remina
    # unchanged. Construct phasors of the nominated reference antenna
    ref_g_x = complex_gains[ref_ant, :, 0]
    ref_g_x = ref_g_x / np.abs(ref_g_x)

    ref_g_y = complex_gains[ref_ant, :, 3]
    ref_g_y = ref_g_y / np.abs(ref_g_y)

    # Now here is the math, from one Captain Daniel Mitchell
    # g_x and g_y.d_yx by g_x(ref) and g_y and g_x.d_xy by g_y(ref).
    # i.e. assuming that xy-phase = 0 (due to the ODC) and that the cross terms are leakage.
    # Since calibrate solves for the Jones directly, the off-diagonals are already
    # multiplied through by the appropriate g_y and g_x.
    g_x_prime = g_x / ref_g_x
    g_xy_prime = g_xy / ref_g_y  # Leakage of y into x, so reference the y
    g_yx_prime = g_yx / ref_g_x  # Leakage of x into y, so reference the x
    g_y_prime = g_y / ref_g_y

    # Construct the output array to slice things into
    bp_p = np.zeros_like(complex_gains) * np.nan

    # Place things into place
    bp_p[:, :, 0] = g_x_prime
    bp_p[:, :, 1] = g_xy_prime
    bp_p[:, :, 2] = g_yx_prime
    bp_p[:, :, 3] = g_y_prime

    return bp_p


def divide_bandpass_by_ref_ant(complex_gains: np.ndarray, ref_ant: int) -> np.ndarray:
    """Divide the bandpass complex gains (solved for initially by something like
    calibrate) by a nominated reference antenna. In the case of ``calibrate``
    there is no implicit reference antenna.

    The input complex gains should be in the form:
    >> (ant, channel, pol)

    Internally this function will construct a phasor:
    >> phasor = Jones_{ref_ant} / abs(Jones_{ref_ant})
    >> shift = Jones_{ref_ant}[0] / and(Jones_{ref_ant}[0])
    >> phasor_shifted = phasor / shift

    which is applied to all antennas in ``complex_gains``. The resulting
    solutions will all have been referenced to the `G_x` of the reference
    antenna. In other words, the phase of all `G_x` items of the reference
    antenna will be zero.

    Args:
        complex_gains (np.ndarray): The complex gains that will be normalised
        ref_ant (int): The desired reference antenna to use

    Returns:
        np.ndarray: The normalised bandpass solutions
    """
    assert len(complex_gains.shape) == 3, (
        f"The shape of the input complex gains should be of rank 3 in form (ant, chan, pol). Received {complex_gains.shape}"
    )

    logger.info(
        f"Dividing bandpass gain solutions using reference antenna={ref_ant} with shifted phasor"
    )

    # Make a copy of the data to avoid editing it wherever else it might be.
    # Trust nothing you pirate.
    complex_gains = complex_gains.copy()

    ref_ant_solutions = complex_gains[ref_ant]

    ref_ant_phasor = ref_ant_solutions / np.abs(ref_ant_solutions)
    ref_ant_shift = (ref_ant_solutions[:, 0] / np.abs(ref_ant_solutions[:, 0]))[:, None]

    # This preserves the consistency of the phase within an single Jones. Essentially
    # setting the reference to the Xx term of the reference antenna.
    phasor_shift = ref_ant_phasor / ref_ant_shift

    logger.info("Multipply, no divide")
    complex_gains = (
        complex_gains / ref_ant_phasor[None, :, :] * phasor_shift[None, :, :]
    )

    return complex_gains


def smooth_data(
    data: np.ndarray,
    window_size: int,
    polynomial_order: int,
    apply_median_filter: bool = False,
) -> np.ndarray:
    """Smooth a 1-dimensional dataset. Internally it uses a savgol filter as
    implemented in scipy.signal.savgol_filter. It is intended to be used to
    smooth the real and imaginary components of the complex gains of the
    bandpass solutions.

    Datapoints that are NaN's are first filled by linearly interpolation the
    closest valid data points. Once the savgol filter has been applied these
    datapoints are then remasked with a NaN.

    If ``median_filter`` is ``True`` then the raw data (without any interpolation)
    will first be passed through a median boxcar filter with a window size of
    ``window_size``.

    Args:
        data (np.ndarray): The 1-dimensional data to be smoothed.
        window_size (int): The size of the window function of the savgol filter. Passed directly to savgol.
        polynomial_order (int): The order of the polynomial of the savgol filter. Passed directly to savgol.
        apply_median_filter (bool, optional): Apply a median filter to the data before applying the savgol filter using the same window size. Defaults to False.

    Returns:
        np.ndarray: Smoothed dataset
    """

    # Make a copy so we do not mess around with the original numpy data
    # where ever it might be. Trust nothing you sea dog.
    data = data.copy()

    if np.all(~np.isfinite(data)):
        return data

    if apply_median_filter:
        data = median_filter(input=data, size=window_size)

    # Before we smooth we need to fill in channels that are flagged with nans.
    # For this we will apply a simply linear interpolation across the blanked
    # regions, smooth, and then reflag them later
    x = np.arange(len(data))
    mask = ~np.isfinite(data)
    d_interp = np.interp(x[mask], x[~mask], data[~mask])

    # This is the filling in of the blanks
    data[mask] = d_interp

    # Now we smooth. This savgol filter fits a polynomial to successive subsets of data
    # in a manner similar to a box car. The final positional argument here denoted the
    # behaviour of the edge where the window (second positional argument) does not have
    # enough data. This process is similar to the original implemented in bptools, except
    # here we are using a polynomial, not a set of harmonic basis functions.
    smoothed_data = savgol_filter(data, window_size, polynomial_order)
    smoothed_data[mask] = np.nan

    return smoothed_data


def smooth_bandpass_complex_gains(
    complex_gains: np.ndarray,
    window_size: int = 16,
    polynomial_order: int = 4,
    apply_median_filter: bool = True,
    smooth_jones_elements: tuple[int, ...] = (0, 1, 2, 3),
) -> np.ndarray:
    """Smooth bandpass solutions by applying a savgol filter to the real and imaginary components
    of each of the antenna based polarisation solutions across channels.

    Experience suggests that this performs better once the data have been divided by a reference
    antenna. Flagged channels (represented by NaNs) will be maintained in the smoothed
    output data.

    The input bandpass data contained by ``complex_gains`` is expected to be in the form:
    > [ants, chans, pols]

    If ``median_filter`` is ``True`` then the raw data (without any interpolation)
    will first be passed through a median boxcar filter with a window size of
    ``window_size``.

    Args:
        complex_gains (np.ndarray): Data to be smoothed.
        window_size (int, optional): The size of the window function of the savgol filter. Passed directly to savgol. Defaults to 16.
        polynomial_order (int, optional): The order of the polynomial of the savgol filter. Passed directly to savgol. Defaults to 4.
        apply_median_filter (bool, optional): Apply a median filter to the data before applying the savgol filter using the same window size. Defaults to True.
        smoothe_jones_elements (Tuple[int, ...], optional): Which elements of the antennae Jones will be smoothed through frequency, i.e. X_x, X_y, Y_x, Y_y. Defaults to (0, 1, 2, 3).

    Returns:
        np.ndarray: Smoothed complex gains
    """

    assert len(complex_gains.shape) == 3, (
        f"The shape of the input complex gains should be of rank 3 in form (ant, chan, pol). Received {complex_gains.shape}"
    )

    # Duplicate the original, ya filthy pirate
    smoothed_complex_gains = complex_gains.copy()

    ants, chans, pols = complex_gains.shape

    logger.info(f"Smoothing using {window_size=} {polynomial_order=}")

    for ant in range(ants):
        # TODO: This will be smoothing the X_y and Y_x. Should this actually be done?
        for pol in smooth_jones_elements:
            assert pol in (0, 1, 2, 3), f"{pol=} is not valid Jones entry. "

            smoothed_complex_gains[ant, :, pol].real = smooth_data(
                data=complex_gains[ant, :, pol].real,
                window_size=window_size,
                polynomial_order=polynomial_order,
                apply_median_filter=apply_median_filter,
            )
            smoothed_complex_gains[ant, :, pol].imag = smooth_data(
                data=complex_gains[ant, :, pol].imag,
                window_size=window_size,
                polynomial_order=polynomial_order,
                apply_median_filter=apply_median_filter,
            )

    return smoothed_complex_gains
