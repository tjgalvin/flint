"""Tooling to attempt to flag components related to bandpasses. The idea
being that poorly calibration channels in the antenna-based gains should
be removed. 

At this point there are no attempts to smooth or interpolate these flagged
components of the bandpass. 
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from flint.logging import logger


def complex_gain_model(
    xs: np.ndarray, gradient: float, phase_offset: float
) -> np.ndarray:
    """Simulate a simple set of complex gains. No consideration made to amplitudes,
    only considering phase

    Args:
        xs (np.ndarray): Positions to evaluate model at.
        gradient (float): Gradient of the phase-slope, rate of wrapping
        phase_offset (float): Initial phasor offset of the model

    Returns:
        np.ndarray: Equal length to input xs of complex numbers representing phase-ramp
    """

    gains = np.exp(-1j * 2.0 * np.pi * gradient * xs + phase_offset * 1j)
    return gains


def fit_complex_gain_model(*args):
    """A fudge to help the curve_fit along."""
    return np.angle(complex_gain_model(*args))


def flag_outlier_phase(complex_gains: np.ndarray, flag_cut: float) -> np.ndarray:
    """This procedure attempts to identify channels in the bandpass solutions to
    flag but searching for gains with outlier phases. Typically, ASKAP solutions
    have a phase-slope across the band (i.e. a delay). This phase-slope first
    needs to be 'unwrapped' in order to correctly identify outliers reliably.

    Internally this function constructs an initial model of the unknown phase-slope.
    It estimates this by looking at the bulk set of gradients on a channel-to-channel
    basis, and looking at the builk distribution after removing large jumps (possibly
    RFI, possible a wrap). The initial phase offset is taken as the phase of the first
    valid gain.

    The initial model is used to unwrap the data, allowing a lest-squares fitter to
    more reliably fit. Once the fitter has been executed, the final cuts are applied
    against the unwrapped phase residuals.

    Experience shows that best results are achieved when the input complex-gains
    have been normalised against a reference antenna. There may be complex structures
    when the raw antenna phase vs frequency plots that can not be reliably fit for
    in this manner. These structures (I believe) arise from the beam-wise spectral
    sub-windows. See BPTools for a more thorough explanation.

    Args:
        complex_gains (np.ndarray): The comples-gains as a function of frequency.
        flag_cut (float): The significance a point should be before flagged as outlier

    Returns:
        np.ndarray: Array of equal length to complex_gains indicating channels to flag
    """

    idxs = np.arange(complex_gains.shape[0])

    # Step one: attempt to guess initial conditions of model to unwrap.
    # These calibration solutions typically have a unknown delay and phase
    # offset. For the least-squares fitter to fit the data robustly, decent
    # initial guesses are needed.
    complex_mask = np.isfinit(complex_gains)
    gain_angles = np.angle(complex_gains)
    init_phase_offset = gain_angles[complex_mask][0]

    # dividing by run is not strictly needed at the moment
    init_gradients = (gain_angles[1:] - gain_angles[:-1]) / (idxs[1:] - idxs[:-1])
    # The second half of this mask is to capture and exclude moments where the phase slop wraps.
    # The initial guess just needs to be in the ball park.
    init_gradients_mask = np.isfinite(init_gradients) & (
        np.abs(init_gradients) < np.pi / 2
    )
    init_gradient = np.median(init_gradients[init_gradients_mask])
    # TODO: Pretty sure this 2pi factor can be removed if the complex_gains_model
    # also has its 2pi removed. It feels like I am missing a something here.
    init_p0 = (init_gradient / (2.0 * np.pi), init_phase_offset)

    # Now construct the initial guess model, used to unwrap the data
    unwrapped_complex_gains = complex_gains / complex_gain_model(idxs, *init_p0)

    # Since there should be a fairly decent initial unwrapped with
    # an additional additive offset to set the bulk of the phases
    # to near zero, we can pass the fitter a fairly simple guesses
    p0 = [0, 0]
    results = curve_fit(
        fit_complex_gain_model,
        idxs[complex_mask],
        np.angle(unwrapped_complex_gains)[complex_mask],
        p0,
    )

    # Make the residuals
    unwrapped_residuals = np.angle(
        unwrapped_complex_gains / complex_gain_model(idxs, *results[0])
    )

    # Apply the final cuts to identify channels of excess phase offset, indicating
    # RFI.
    # TODO: Use more robust statistics, like MAD
    unwrapped_residual_mean = np.nanmean(unwrapped_residuals)
    unwrapped_residual_std = np.nanstd(unwrapped_residuals)

    final_mask = np.isfinite(unwrapped_residuals) & (
        np.abs(unwrapped_residuals)
        < (unwrapped_residual_mean + flag_cut * unwrapped_residual_std)
    )

    return ~final_mask
