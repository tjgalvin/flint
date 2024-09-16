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
    _create_bound_box_plane,
    _get_alpha_linmos_option,
    _get_holography_linmos_options,
    _get_image_weight_plane,
    create_bound_box,
    generate_weights_list_and_files,
    trim_fits_image,
)


def test_get_image_weight_plane():
    """The extraction of weights per plane"""
    data = np.arange(100).reshape((10, 10))

    with pytest.raises(AssertionError):
        _get_image_weight_plane(image_data=data, mode="noexists")  # type: ignore

    assert np.isclose(
        0.0016, _get_image_weight_plane(image_data=data, mode="mad"), atol=0.0001
    )
    assert np.isclose(
        0.00120012, _get_image_weight_plane(image_data=data, mode="std"), atol=0.0001
    )


def create_fits_image(out_path, image_size=(1000, 1000)):
    data = np.zeros(image_size)
    data[10:600, 20:500] = 1
    data[data == 0] = np.nan

    header = fits.header.Header({"CRPIX1": 10, "CRPIX2": 20})

    fits.writeto(out_path, data=data, header=header)


def create_image_cube(out_path):
    data = np.arange(20 * 100).reshape((20, 10, 10))
    header = fits.header.Header({"CRPIX1": 10, "CRPIX2": 20, "CRPIX3": 1})

    fits.writeto(out_path, header=header, data=data)


def test_linmos_alpha_option():
    """Ensure the rotation string supplied to linmos is calculated appropriately"""

    options_str = _get_alpha_linmos_option(pol_axis=None)
    assert options_str == ""

    options_str = _get_alpha_linmos_option(pol_axis=np.deg2rad(-45))
    expected_str = "linmos.primarybeam.ASKAP_PB.alpha = 0.0 # in radians\n"
    assert options_str == expected_str

    with pytest.raises(AssertionError):
        _get_alpha_linmos_option(pol_axis=1234)


def test_get_image_weights(tmpdir):
    """See whether the weights computed per plane in a cube work appropriately"""
    cube_weight = Path(tmpdir) / "cubeweight"
    cube_weight.mkdir(parents=True, exist_ok=True)
    cube_fits = cube_weight / "cube.fits"

    create_image_cube(out_path=cube_fits)
    weight_file = cube_fits.with_suffix(".weights.txt")
    assert not weight_file.exists()

    generate_weights_list_and_files(image_paths=[cube_fits], mode="mad")
    assert weight_file.exists()
    lines = weight_file.read_text().split("\n")
    assert len(lines) == 21, f"{lines}"


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


def test_trim_fits_cube(tmp_path):
    """Ensure that fits files that has cube can be trimmed appropriately based on row/columns with valid pixels"""
    tmp_dir = tmp_path / "cube"
    tmp_dir.mkdir()

    out_fits = tmp_dir / "example.fits"

    cube_size = (12, 1000, 1000)
    data = np.zeros(cube_size)
    data[3, 10:600, 20:500] = 1
    data[data == 0] = np.nan

    header = fits.header.Header({"CRPIX1": 10, "CRPIX2": 20})

    fits.writeto(out_fits, data=data, header=header)

    og_hdr = fits.getheader(out_fits)
    assert og_hdr["CRPIX1"] == 10
    assert og_hdr["CRPIX2"] == 20

    trim_fits_image(out_fits)
    trim_hdr = fits.getheader(out_fits)
    trim_data = fits.getdata(out_fits)
    assert trim_hdr["CRPIX1"] == -10
    assert trim_hdr["CRPIX2"] == 10
    assert trim_data.shape == (12, 589, 479)  # type: ignore


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
    """Create a bounding box around a region of valid non-nan/inf pixels"""
    data = np.zeros((1000, 1000))
    data[10:600, 20:500] = 1
    data[data == 0] = np.nan

    bb = create_bound_box(image_data=data)

    assert isinstance(bb, BoundingBox)
    assert bb.xmin == 10
    assert bb.xmax == 599  # slices upper limit is not inclusive
    assert bb.ymin == 20
    assert bb.ymax == 499  # slices upper limit no inclusive


def test_bounding_box_none():
    """Return None if there are no valid pixels to create a bounding box around"""
    data = np.zeros((1000, 1000)) * np.nan

    bb = _create_bound_box_plane(image_data=data)
    assert bb is None

    bb = create_bound_box(image_data=data)
    assert isinstance(bb, BoundingBox)
    assert bb.xmin == 0
    assert bb.xmin == 0
    assert bb.xmax == 999
    assert bb.ymax == 999


def test_bounding_box_cube():
    """Cube cut bounding boxes."""
    data = np.zeros((3, 1000, 1000))
    data[:, 10:600, 20:500] = 1
    data[data == 0] = np.nan

    with pytest.raises(AssertionError):
        _create_bound_box_plane(image_data=data)

    bb = create_bound_box(image_data=data)
    assert isinstance(bb, BoundingBox)
    assert bb.xmin == 10
    assert bb.xmax == 599
    assert bb.ymin == 20
    assert bb.ymax == 499


def test_bounding_box_cube_different_bounds():
    """Cube cut bounding boxes, where the largest bounding box that
    captures all valid pixels"""
    data = np.zeros((3, 1000, 1000))
    data[0, 10:600, 20:500] = 1
    data[1, 100:200, 600:800] = 1
    data[2, 800:888, 20:500] = 1

    data[data == 0] = np.nan

    with pytest.raises(AssertionError):
        _create_bound_box_plane(image_data=data)

    bb = create_bound_box(image_data=data)
    assert isinstance(bb, BoundingBox)
    assert bb.xmin == 10
    assert bb.xmax == 887
    assert bb.ymin == 20
    assert bb.ymax == 799


def test_bounding_box_with_mask():
    """Create a bounding box where the input is converted to a boolean arrau"""
    data = np.zeros((1000, 1000))
    data[10:600, 20:500] = 1

    bb = create_bound_box(image_data=data, is_masked=True)

    assert isinstance(bb, BoundingBox)
    assert bb.xmin == 10
    assert bb.xmax == 599  # slices upper limit is not inclusive
    assert bb.ymin == 20
    assert bb.ymax == 499  # slices upper limit no inclusive
