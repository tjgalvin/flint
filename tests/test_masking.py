from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from flint.masking import create_snr_mask_from_fits, MaskingOptions
from flint.naming import FITSMaskNames

SHAPE = (100, 100)


@pytest.fixture
def fits_dir(tmpdir):
    fits_dir = Path(tmpdir) / "fits"
    fits_dir.mkdir()

    shape = SHAPE

    img_data = np.ones(shape)
    bkg_data = np.ones(shape) * 0.5
    rms_data = np.ones(shape) * 0.1

    fits.writeto(fits_dir / "image.fits", img_data)
    fits.writeto(fits_dir / "rms.fits", rms_data)
    fits.writeto(fits_dir / "bkg.fits", bkg_data)

    return fits_dir


def test_make_masking_options():
    """Just a dump test to make sure the options structure is ok"""

    masking_options = MaskingOptions()

    assert masking_options.base_snr_clip != -1

    masking_options = masking_options.with_options(base_snr_clip=-1)
    assert masking_options.base_snr_clip == -1


def test_fits_masking(fits_dir):
    masking_options = MaskingOptions(flood_fill=False)
    names = create_snr_mask_from_fits(
        fits_image_path=fits_dir / "image.fits",
        fits_rms_path=fits_dir / "rms.fits",
        fits_bkg_path=fits_dir / "bkg.fits",
        masking_options=masking_options,
    )

    assert isinstance(names, FITSMaskNames)
    assert names.mask_fits.exists()
    assert names.signal_fits is None

    mask_data = fits.getdata(names.mask_fits)
    valid = np.sum(mask_data)
    assert valid == np.prod(SHAPE)


def test_fits_masking_with_signal(fits_dir):
    masking_options = MaskingOptions(flood_fill=False)
    names = create_snr_mask_from_fits(
        fits_image_path=fits_dir / "image.fits",
        fits_rms_path=fits_dir / "rms.fits",
        fits_bkg_path=fits_dir / "bkg.fits",
        masking_options=masking_options,
        create_signal_fits=True,
    )

    assert isinstance(names, FITSMaskNames)
    assert names.mask_fits.exists()
    assert names.signal_fits.exists()

    mask_data = fits.getdata(names.mask_fits)
    valid = np.sum(mask_data)
    assert valid == np.prod(SHAPE)

    signal_data = fits.getdata(names.signal_fits)
    assert np.allclose(signal_data, 5)
