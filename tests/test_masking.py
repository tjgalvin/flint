from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from flint.masking import (
    MaskingOptions,
    _verify_set_positive_seed_clip,
    beam_shape_erode,
    consider_beam_mask_round,
    create_beam_mask_kernel,
    create_snr_mask_from_fits,
    minimum_boxcar_artefact_mask,
)
from flint.naming import FITSMaskNames

SHAPE = (100, 100)


def test_create_beam_mask_kernel():
    """See whether the beam kernel creation mask can return a correct mask"""
    fits_header = fits.Header(
        dict(
            CDELT1=-0.000694444444444444,
            CDELT2=0.000694444444444444,
            BMAJ=0.00340540107886635,
            BMIN=0.00283268735470751,
            BPA=74.6618858613889,
        )
    )
    mask_1 = create_beam_mask_kernel(fits_header=fits_header)
    assert mask_1.shape == (100, 100)
    assert np.sum(mask_1) == 12

    mask_2 = create_beam_mask_kernel(fits_header=fits_header, minimum_response=0.1)
    assert mask_2.shape == (100, 100)
    assert np.sum(mask_2) == 52

    with pytest.raises(AssertionError):
        fits_header = fits.Header(
            dict(
                CDELT1=-0.000694444444444444,
                CDELT2=0.2000694444444444444,
                BMAJ=0.00340540107886635,
                BMIN=0.00283268735470751,
                BPA=74.6618858613889,
            )
        )
        create_beam_mask_kernel(fits_header=fits_header)

    with pytest.raises(KeyError):
        fits_header = fits.Header(
            dict(
                CDELT1=-0.000694444444444444,
                BMAJ=0.00340540107886635,
                BMIN=0.00283268735470751,
                BPA=74.6618858613889,
            )
        )
        create_beam_mask_kernel(fits_header=fits_header)


def test_beam_shape_erode():
    """Ensure that the beam shape erosion approach works. This should drop out pixels
    should the beam shape structure connectivity be statisifed"""
    fits_header = fits.Header(
        dict(
            CDELT1=-0.000694444444444444,
            CDELT2=0.000694444444444444,
            BMAJ=0.00340540107886635,
            BMIN=0.00283268735470751,
            BPA=74.6618858613889,
        )
    )

    mask = np.zeros((500, 500)).astype(bool)

    mask[300, 300] = True
    assert np.sum(mask) == 1

    new_mask = beam_shape_erode(mask=mask, fits_header=fits_header)
    assert np.sum(new_mask) == 0

    mask = np.zeros((200, 200)).astype(bool)
    mask[100:130, 100:130] = True
    assert np.sum(mask) == 900
    erode_mask = beam_shape_erode(mask=mask, fits_header=fits_header)
    assert np.sum(erode_mask) == 729


def test_consider_beam_masking_round():
    """Test to ensure the beam mask consideration log is correct"""
    lower = ("all", "ALL", "aLl")
    states = (
        consider_beam_mask_round(current_round=1, mask_rounds=low) for low in lower
    )

    assert all(states)

    assert consider_beam_mask_round(current_round=3, mask_rounds=1)
    assert not consider_beam_mask_round(current_round=0, mask_rounds=1)

    assert consider_beam_mask_round(current_round=3, mask_rounds=(1, 2, 3, 4, 5))
    assert not consider_beam_mask_round(current_round=3, mask_rounds=(1, 2, 4, 5))

    assert not consider_beam_mask_round(current_round=3, mask_rounds=None)

    assert not consider_beam_mask_round(
        current_round=3, mask_rounds=1, allow_beam_masks=False
    )
    assert consider_beam_mask_round(
        current_round=3, mask_rounds=1, allow_beam_masks=True
    )
    assert consider_beam_mask_round(current_round=1, mask_rounds=1)


def test_minimum_boxcar_artefact():
    """See if the minimum box care artefact suppressor can suppress the
    bright artefact when a bright negative artefact
    """
    img = np.zeros((SHAPE))

    img[30:40, 30:40] = 10
    img_mask = img > 5

    out_mask = minimum_boxcar_artefact_mask(
        signal=img, island_mask=img_mask, boxcar_size=10
    )
    assert np.all(img_mask == out_mask)
    assert img_mask is not out_mask

    img[41:45, 30:40] = -20
    out_mask = minimum_boxcar_artefact_mask(
        signal=img, island_mask=img_mask, boxcar_size=10
    )
    assert not np.all(img_mask == out_mask)


def test_minimum_boxcar_artefact_blanked():
    """See if the minimum box care artefact suppressor can suppress the
    bright artefact when a bright negative artefact
    """
    img = np.zeros((SHAPE))

    img[30:40, 30:40] = 10
    img[41:45, 30:40] = -20

    img_mask = img > 5

    out_mask = minimum_boxcar_artefact_mask(
        signal=img, island_mask=img_mask, boxcar_size=10, increase_factor=1000
    )
    assert out_mask is not img_mask
    assert np.all(out_mask[30:40, 30:40] == False)  # noqa


def test_minimum_boxcar_large_bright_island():
    """This one checks to make sure that if the boxcar is smaller than
    a positive islane that the island is not the island_min
    """

    img = np.zeros(SHAPE)
    img[30:40, 30:40] = 10
    img_mask = img > 5

    out_mask = minimum_boxcar_artefact_mask(
        signal=img, island_mask=img_mask, boxcar_size=2
    )
    assert np.all(img_mask == out_mask)


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


def test_verify_set_seed_clip():
    """Make sure the flood seed clip handles items above all possible values"""
    signal = np.ones((100, 100)) * 10.0

    flood_clip = _verify_set_positive_seed_clip(signal=signal, positive_seed_clip=9.0)

    assert flood_clip == 9.0
    flood_clip = _verify_set_positive_seed_clip(
        signal=signal, positive_seed_clip=999999.0
    )
    assert flood_clip == 9.0


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
