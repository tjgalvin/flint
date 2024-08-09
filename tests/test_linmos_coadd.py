"""Tests related to components in the yandasoft linmos coadd.
At the moment this is not testing the actual application. Just
some of the helper functions around it.
"""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from flint.coadd.linmos import BoundingBox, create_bound_box, trim_fits_image


def create_fits_image(out_path, image_size=(1000, 1000)):
    data = np.zeros(image_size)
    data[10:600, 20:500] = 1
    data[data == 0] = np.nan

    header = fits.header.Header({"CRPIX1": 10, "CRPIX2": 20})

    fits.writeto(out_path, data=data, header=header)


def test_trim_fits(tmp_path):
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
