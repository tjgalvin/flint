"""Basic tests for utility functions"""

from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

from flint.utils import estimate_skycoord_centre, get_packaged_resource_path


def test_package_resource_path_folder():
    dir_path = get_packaged_resource_path(package="flint.data", filename="")

    assert isinstance(dir_path, Path)
    assert dir_path.exists()


def test_package_resource_path_askap_lua():
    askap_lua = get_packaged_resource_path(
        package="flint.data.aoflagger", filename="ASKAP.lua"
    )

    assert isinstance(askap_lua, Path)
    assert askap_lua.exists()

    with open(askap_lua, "r") as open_lua:
        line = open_lua.readline()
        assert line == "--[[\n"


def test_package_resource_path_skymodel():
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
    ras = np.arange(-3, 3, 1) + 180.0
    decs = np.arange(-3, 3, 1) - 40.0

    sky_pos = SkyCoord(ras, decs, unit=(u.deg, u.deg))

    mean_pos = estimate_skycoord_centre(sky_positions=sky_pos)

    print(mean_pos)

    assert np.isclose(mean_pos.ra.deg, 179.54350474)
    assert np.isclose(mean_pos.dec.deg, -40.51256163)


def test_estimate_skycoord_centre_wrap():
    ras = np.arange(-3, 3, 1) + 360.0 % 360
    decs = np.arange(-3, 3, 1) - 40.0

    sky_pos = SkyCoord(ras, decs, unit=(u.deg, u.deg))

    mean_pos = estimate_skycoord_centre(sky_positions=sky_pos)

    print(mean_pos)

    assert np.isclose(mean_pos.ra.deg, 359.54349533)
    assert np.isclose(mean_pos.dec.deg, -40.51255648)
