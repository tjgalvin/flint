"""Basic tests for utility functions"""

import pytest
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits

from flint.utils import (
    estimate_skycoord_centre,
    get_packaged_resource_path,
    generate_stub_wcs_header,
)


@pytest.fixture
def rms_path(tmpdir):
    rms_path = Path(
        get_packaged_resource_path(
            package="flint.data.tests",
            filename="SB39400.RACS_0635-31.beam0-MFS-subimage_rms.fits",
        )
    )

    return rms_path


def test_wcs_getter():
    """Make a basic wcs object"""
    # TODO: Need some proper tests here. Translate to sky positions etc
    w = generate_stub_wcs_header(
        ra=180, dec=-45, image_shape=(8000, 8000), pixel_scale=0.01
    )

    assert isinstance(w, WCS)


def test_wcs_getter_quantity():
    """Make a basic wcs object that includes different pixel types"""
    w = generate_stub_wcs_header(
        ra=180, dec=-45, image_shape=(8000, 8000), pixel_scale=0.01 * u.deg
    )
    assert isinstance(w, WCS)

    w = generate_stub_wcs_header(
        ra=180, dec=-45, image_shape=(8000, 8000), pixel_scale="2.5arcsec"
    )
    assert isinstance(w, WCS)


def test_wcs_getter_withbase(rms_path):
    """Make a wcs object overriding the wcs from an existing fits file"""
    hdr = fits.getheader(rms_path)
    w = generate_stub_wcs_header(
        ra=180,
        dec=-45,
        image_shape=(8000, 8000),
        pixel_scale=0.01,
        base_wcs=WCS(hdr),
    )

    assert isinstance(w, WCS)

    w2 = generate_stub_wcs_header(
        ra=180,
        dec=-45,
        image_shape=(8000, 8000),
        pixel_scale=0.01,
        base_wcs=rms_path,
    )

    assert isinstance(w2, WCS)


def test_package_resource_path_folder():
    """Ensure the utils package path resource getter works"""
    dir_path = get_packaged_resource_path(package="flint.data", filename="")

    assert isinstance(dir_path, Path)
    assert dir_path.exists()


def test_package_resource_path_askap_lua():
    """Ensure the utils package path resource getter work, and check the contents of a file"""
    askap_lua = get_packaged_resource_path(
        package="flint.data.aoflagger", filename="ASKAP.lua"
    )

    assert isinstance(askap_lua, Path)
    assert askap_lua.exists()

    with open(askap_lua, "r") as open_lua:
        line = open_lua.readline()
        assert line == "--[[\n"


def test_package_resource_path_skymodel():
    """Ensure the utils package path resource getter work, and check the contents of a file"""
    askap_model = get_packaged_resource_path(
        package="flint.data.models", filename="1934-638.calibrate.txt"
    )

    assert isinstance(askap_model, Path)
    assert askap_model.exists()

    with open(askap_model, "r") as open_model:
        line = open_model.readline()
        assert (
            line
            == "Format = Name, Type, Ra, Dec, I, SpectralIndex, LogarithmicSI, ReferenceFrequency='888500000.0', MajorAxis, MinorAxis, Orientation\n"
        )


def test_estimate_skycoord_centre():
    """Estimate the centre position from a collection of sky positions"""
    ras = np.arange(-3, 3, 1) + 180.0
    decs = np.arange(-3, 3, 1) - 40.0

    sky_pos = SkyCoord(ras, decs, unit=(u.deg, u.deg))

    mean_pos = estimate_skycoord_centre(sky_positions=sky_pos)

    print(mean_pos)

    assert np.isclose(mean_pos.ra.deg, 179.54350474)
    assert np.isclose(mean_pos.dec.deg, -40.51256163)


def test_estimate_skycoord_centre_wrap():
    """Estimate the mean center sky position that wraps around 360 -> 0 degrees in ra"""
    ras = np.arange(-3, 3, 1) + 360.0 % 360
    decs = np.arange(-3, 3, 1) - 40.0

    sky_pos = SkyCoord(ras, decs, unit=(u.deg, u.deg))

    mean_pos = estimate_skycoord_centre(sky_positions=sky_pos)

    print(mean_pos)

    assert np.isclose(mean_pos.ra.deg, 359.54349533)
    assert np.isclose(mean_pos.dec.deg, -40.51255648)
