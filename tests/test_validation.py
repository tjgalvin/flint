"""Items related to test functions in the validation stage of flint
"""
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pkg_resources
import pytest

from flint.validation import (
    RMSImageInfo,
    get_parser,
    get_rms_image_info,
    calculate_area_correction_per_flux,
)


def test_get_parser():
    get_parser()


@pytest.fixture
def rms_path(tmpdir):
    rms_path = Path(
        pkg_resources.resource_filename(
            "flint", "data/tests/SB39400.RACS_0635-31.beam0-MFS-subimage_rms.fits"
        )
    )

    return rms_path


def test_calculate_area_correction(rms_path):
    """This is not an entirely robust check as the rms image is only
    10 x 15 pixels. More testing for errors in the calling."""

    flux_bin_centres = np.arange(0.00004, 0.01, 0.0001)

    area_frac = calculate_area_correction_per_flux(
        rms_image_path=rms_path, flux_bin_centre=flux_bin_centres, sigma=3
    )

    test = np.ones(100)
    test[:5] = 0.0

    assert np.allclose(test, area_frac)


def test_rms_image_info(rms_path):
    rms_info = get_rms_image_info(rms_path=rms_path)

    assert isinstance(rms_info, RMSImageInfo)
    assert rms_info.path == rms_path
    assert rms_info.no_valid_pixels == 150
    assert rms_info.shape == (10, 15)
    assert np.isclose(0.0001515522, rms_info.median)
    assert np.isclose(0.00015135764, rms_info.minimum)
    assert np.isclose(0.0001518184, rms_info.maximum)
    assert np.isclose(1.1098655e-07, rms_info.std)


def test_expected_namedtuple_get():
    """This is a simple test to ensure the behavour of
    NamedTuple.__getattribute__ remains OK. This is currently
    used in the validation plotting to iterate over known
    surveys
    """

    class Example(NamedTuple):
        a: int
        b: str
        c: float

    test = Example(a=1, b="123", c=1.23)

    assert test.__getattribute__("a") == 1
    assert test.__getattribute__("b") == "123"
    assert test.__getattribute__("c") == 1.23
