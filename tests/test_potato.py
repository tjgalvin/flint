"""Some basic checks around the potato peel functionality"""

import pytest
import shutil
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.table import Table

from flint.imager.wsclean import WSCleanOptions
from flint.peel.potato import (
    find_sources_to_peel,
    load_known_peel_sources,
    PotatoConfigOptions,
    _potato_config_command,
)
from flint.sky_model import (
    generate_pb,
    GaussianResponse,
    SincSquaredResponse,
    AiryResponse,
)
from flint.utils import get_packaged_resource_path


@pytest.fixture
def ms_example(tmpdir):
    ms_zip = Path(
        get_packaged_resource_path(
            package="flint.data.tests",
            filename="SB39400.RACS_0635-31.beam0.small.ms.zip",
        )
    )
    outpath = Path(tmpdir) / "39400"

    shutil.unpack_archive(ms_zip, outpath)

    ms_path = Path(outpath) / "SB39400.RACS_0635-31.beam0.small.ms"

    return ms_path


def test_potato_config_command():
    a = PotatoConfigOptions()
    ex = Path("This/example/SB1234.potato.config")

    command = _potato_config_command(config_path=ex, potato_config_options=a)
    expected = "peel_configuration.py This/example/SB1234.potato.config --image_size 6148 --image_scale 0.0006944 --image_briggs -1.5 --image_channels 4 --image_minuvl 0.0 --peel_size 1000 --peel_scale 0.0006944 --peel_channels 16 --peel_nmiter 7 --peel_minuvl 0.0 --peel_multiscale "

    assert command == expected


def test_check_sources_to_peel(ms_example):
    """See whether the peeling constraints work"""
    image_options = WSCleanOptions(size=8000, scale="2.5arcsec")

    sources = find_sources_to_peel(ms=ms_example, image_options=image_options)

    known_sources = load_known_peel_sources()

    assert len(known_sources) > len(sources)


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
        [0.96701134, 0.96020018, 0.95278619, 0.94478158, 0.93619952],
    )

    assert np.allclose(
        pb_gauss.atten[140:145],
        [0.0006315, 0.0005682, 0.00051086, 0.00045896, 0.00041203],
    )
    assert np.allclose(
        pb_sinc.atten[140:145],
        [0.04699694, 0.04675459, 0.04641816, 0.04599033, 0.04547408],
    )
    assert np.allclose(
        pb_airy.atten[140:145],
        [0.01749399, 0.01748965, 0.01743916, 0.01734373, 0.01720478],
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
