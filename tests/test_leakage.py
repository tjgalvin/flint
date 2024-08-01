"""Testing components in the leakage creation steps"""

import pytest
from pathlib import Path

import numpy as np
from astropy.wcs import WCS

from flint.leakage import FITSImage, load_fits_image
from flint.utils import get_packaged_resource_path


def test_load_fits_image():
    fits_path = get_packaged_resource_path(
        package="flint.data.tests",
        filename="SB56659.RACS_0940-04.beam17.round3-0000-image.sub.fits",
    )

    assert isinstance(fits_path, Path)
    assert fits_path.exists()

    fits_image = load_fits_image(fits_path=fits_path)

    assert isinstance(fits_image, FITSImage)
    assert fits_image.path == fits_path
    assert isinstance(fits_image.data, np.ndarray)
    assert isinstance(fits_image.wcs, WCS)
    assert isinstance(fits_image.header, dict)

    with pytest.raises(AssertionError):
        load_fits_image(fits_path=fits_path.with_suffix(".jack.sparrow"))
