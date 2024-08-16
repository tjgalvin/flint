"""Tests related to components in the yandasoft linmos coadd.
At the moment this is not testing the actual application. Just
some of the helper functions around it.
"""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from flint.coadd.linmos import (
    BoundingBox,
    _get_alpha_linmos_option,
    _get_holography_linmos_options,
    create_bound_box,
    trim_fits_image,
)


def create_fits_image(out_path, image_size=(1000, 1000)):
    data = np.zeros(image_size)
    data[10:600, 20:500] = 1
    data[data == 0] = np.nan

    header = fits.header.Header({"CRPIX1": 10, "CRPIX2": 20})

    fits.writeto(out_path, data=data, header=header)


def test_linmos_alpha_option():
    """Ensure the rotation string supplied to linmos is calculated appropriately"""

    options_str = _get_alpha_linmos_option(pol_axis=None)
    assert options_str == ""

    options_str = _get_alpha_linmos_option(pol_axis=np.deg2rad(-45))
    expected_str = "linmos.primarybeam.ASKAP_PB.alpha = 0.0 # in radians\n"
    assert options_str == expected_str

    with pytest.raises(AssertionError):
        _get_alpha_linmos_option(pol_axis=1234)


def test_linmos_holo_options(tmpdir):
    holofile = Path(tmpdir) / "testholooptions/holo_file.fits"
    holofile.parent.mkdir(parents=True, exist_ok=True)

    with pytest.raises(AssertionError):
        _get_holography_linmos_options(holofile=holofile, pol_axis=None)

    assert _get_holography_linmos_options(holofile=None, pol_axis=None) == ""

    with holofile.open("w") as f:
        f.write("test")

    parset = _get_holography_linmos_options(holofile=holofile, pol_axis=None)
    assert "linmos.primarybeam      = ASKAP_PB\n" in parset
    assert "linmos.removeleakage    = true\n" in parset
    assert f"linmos.primarybeam.ASKAP_PB.image = {str(holofile.absolute())}\n" in parset
    assert "linmos.primarybeam.ASKAP_PB.alpha" not in parset

    parset = _get_holography_linmos_options(holofile=holofile, pol_axis=np.deg2rad(-45))
    assert "linmos.primarybeam      = ASKAP_PB\n" in parset
    assert "linmos.removeleakage    = true\n" in parset
    assert f"linmos.primarybeam.ASKAP_PB.image = {str(holofile.absolute())}\n" in parset
    assert "linmos.primarybeam.ASKAP_PB.alpha" in parset


def test_trim_fits(tmp_path):
    """Ensure that fits files can be trimmed appropriately based on row/columns with valid pixels"""
    tmp_dir = tmp_path / "image"
    tmp_dir.mkdir()

    out_fits = tmp_dir / "example.fits"

    create_fits_image(out_fits)
    og_hdr = fits.getheader(out_fits)
    assert og_hdr["CRPIX1"] == 10
    assert og_hdr["CRPIX2"] == 20

    trim_fits_image(out_fits)
    trim_hdr = fits.getheader(out_fits)
    trim_data = fits.getdata(out_fits)
    assert trim_hdr["CRPIX1"] == -10
    assert trim_hdr["CRPIX2"] == 10
    assert trim_data.shape == (589, 479)


def test_trim_fits_image_matching(tmp_path):
    """See the the bounding box can be passed through for matching to cutout"""

    tmp_dir = Path(tmp_path) / "image_bb_match"
    tmp_dir.mkdir()

    out_fits = tmp_dir / "example.fits"

    create_fits_image(out_fits)
    og_trim = trim_fits_image(out_fits)

    out_fits2 = tmp_dir / "example2.fits"
    create_fits_image(out_fits2)
    og_hdr = fits.getheader(out_fits2)
    assert og_hdr["CRPIX1"] == 10
    assert og_hdr["CRPIX2"] == 20

    trim_fits_image(image_path=out_fits2, bounding_box=og_trim.bounding_box)
    trim_hdr = fits.getheader(out_fits2)
    trim_data = fits.getdata(out_fits2)
    assert trim_hdr["CRPIX1"] == -10
    assert trim_hdr["CRPIX2"] == 10
    assert trim_data.shape == (589, 479)

    out_fits2 = tmp_dir / "example3.fits"
    create_fits_image(out_fits2, image_size=(300, 300))

    with pytest.raises(ValueError):
        trim_fits_image(image_path=out_fits2, bounding_box=og_trim.bounding_box)


def test_bounding_box():
    data = np.zeros((1000, 1000))
    data[10:600, 20:500] = 1
    data[data == 0] = np.nan

    bb = create_bound_box(image_data=data)

    assert isinstance(bb, BoundingBox)
    assert bb.xmin == 10
    assert bb.xmax == 599  # slices upper limit is not inclusive
    assert bb.ymin == 20
    assert bb.ymax == 499  # slices upper limit no inclusive


def test_bounding_box_with_mask():
    data = np.zeros((1000, 1000))
    data[10:600, 20:500] = 1

    bb = create_bound_box(image_data=data, is_masked=True)

    assert isinstance(bb, BoundingBox)
    assert bb.xmin == 10
    assert bb.xmax == 599  # slices upper limit is not inclusive
    assert bb.ymin == 20
    assert bb.ymax == 499  # slices upper limit no inclusive
