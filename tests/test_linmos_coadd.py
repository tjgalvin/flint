"""Tests related to components in the yandasoft linmos coadd. 
At the moment this is not testing the actual application. Just
some of trhe helper functions around it. 
"""

import numpy as np
from astropy.io import fits

from flint.coadd.linmos import create_bound_box, BoundingBox, trim_fits_image


def create_fits_image(out_path):
    data = np.zeros((1000, 1000))
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
