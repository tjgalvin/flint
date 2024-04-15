"""Some basic checks around the potato peel functionality"""

import pytest

import numpy as np
import astropy.units as u
from astropy.table import Table

from flint.peel.potato import load_known_peel_sources
from flint.sky_model import (
    generate_pb,
    GaussianResponse,
    SincSquaredResponse,
    AiryResponse,
)


def test_load_peel_sources():
    """Ensure we can load in the reference set of sources for peeling"""

    tab = load_known_peel_sources()
    assert isinstance(tab, Table)
    assert len(tab) > 4


def test_pb_models():
    """See if the respones can be loaded and executed"""
    freq = 1.0 * u.GHz
    aperture = 12 * u.m
    offset = np.arange(0, 200) * u.arcmin

    pb_gauss = generate_pb(
        pb_type="gaussian", freqs=freq, aperture=aperture, offset=offset
    )
    pb_sinc = generate_pb(
        pb_type="sincsquared", freqs=freq, aperture=aperture, offset=offset
    )
    pb_airy = generate_pb(pb_type="airy", freqs=freq, aperture=aperture, offset=offset)

    assert isinstance(pb_gauss, GaussianResponse)
    assert isinstance(pb_sinc, SincSquaredResponse)
    assert isinstance(pb_airy, AiryResponse)

    assert np.isclose(pb_gauss.atten[0], 1.0)
    assert np.isclose(pb_sinc.atten[0], 1.0)
    assert np.isclose(pb_airy.atten[0], 1.0)

    for m in (pb_gauss, pb_sinc, pb_airy):
        assert m.freqs == freq

    # Following known values were generated with the following
    # code, verified, and placed here
    """
    freq = 1.0 * u.GHz
    aperture = 12 * u.m
    offset = np.arange(0, 200) * u.arcmin

    pb_gauss = generate_pb(
        pb_type="gaussian", freqs=freq, aperture=aperture, offset=offset
    )
    pb_sinc = generate_pb(
        pb_type="sincsquared", freqs=freq, aperture=aperture, offset=offset
    )
    pb_airy = generate_pb(pb_type="airy", freqs=freq, aperture=aperture, offset=offset)
    """

    assert np.allclose(
        pb_gauss.atten[10:15],
        [0.96310886, 0.95553634, 0.94731093, 0.93845055, 0.9289744],
    )
    assert np.allclose(
        pb_sinc.atten[10:15],
        [0.96516646, 0.95797617, 0.95015023, 0.94170175, 0.93264483],
    )
    assert np.allclose(
        pb_airy.atten[10:15],
        [0.96702892, 0.96022132, 0.95281117, 0.9448107, 0.93623302],
    )


def test_bad_pb_model():
    """Makes sure ValueError raised on unknown model"""
    freq = 1.0 * u.GHz
    aperture = 12 * u.m
    offset = np.arange(0, 200) * u.arcmin

    with pytest.raises(ValueError):
        generate_pb(
            pb_type="ThisDoesNotExist", freqs=freq, aperture=aperture, offset=offset
        )
