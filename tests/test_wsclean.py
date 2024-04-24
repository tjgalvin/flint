"""Testing some wsclean functionality."""

from pathlib import Path

import pytest

from flint.exceptions import CleanDivergenceError
from flint.imager.wsclean import (
    ImageSet,
    get_wsclean_output_names,
    _wsclean_output_callback,
)


def test_wsclean_divergence():
    good = (
        "Iteration 59228, scale 0 px : -862.94 µJy at 3729,3746",
        "Opening reordered part 0 spw 0 for /scratch3/gal16b/flint_peel/40470/SB40470.RACS_1237+00.beam4.round1.ms",
        "Opening reordered part 0 spw 0 for /scratch3/gal16b/flint_peel/40470/SB40470.RACS_1237+00.beam4.round1.ms",
        "Although KJy there is no iterat ion, not the lack of a capital-I and the space, clever pirate",
    )
    for g in good:
        _wsclean_output_callback(line=g)

    bad = "Iteration 59228, scale 0 px : -862.94 KJy at 3729,3746"
    with pytest.raises(CleanDivergenceError):
        _wsclean_output_callback(line=bad)

    with pytest.raises(AssertionError):
        _wsclean_output_callback(line=tuple("A tuple of text".split()))


def test_wsclean_output_named_raises():
    with pytest.raises(FileExistsError):
        _ = get_wsclean_output_names(
            prefix="JackSparrow", subbands=4, verify_exists=True
        )


def test_wsclean_output_named_check_when_adding():
    image_set = get_wsclean_output_names(
        prefix="JackSparrow",
        subbands=4,
        verify_exists=True,
        check_exists_when_adding=True,
    )

    assert isinstance(image_set, ImageSet)
    assert len(image_set.image) == 0


def test_wsclean_output_named():
    image_set = get_wsclean_output_names(prefix="JackSparrow", subbands=4)

    assert isinstance(image_set, ImageSet)
    assert image_set.prefix == "JackSparrow"

    assert image_set.image is not None
    assert len(image_set.image) == 5
    assert isinstance(image_set.image[0], Path)

    assert image_set.dirty is not None
    assert len(image_set.dirty) == 5
    assert isinstance(image_set.dirty[0], Path)

    assert image_set.model is not None
    assert len(image_set.model) == 5
    assert isinstance(image_set.model[0], Path)

    assert image_set.residual is not None
    assert len(image_set.residual) == 5
    assert isinstance(image_set.residual[0], Path)

    assert image_set.psf is not None
    assert len(image_set.psf) == 5
    assert isinstance(image_set.psf[0], Path)


def test_wsclean_output_named_wpols():
    image_set = get_wsclean_output_names(
        prefix="JackSparrow", subbands=4, pols=("I", "Q")
    )

    assert isinstance(image_set, ImageSet)
    assert image_set.prefix == "JackSparrow"

    expected = 10
    assert image_set.image is not None
    assert len(image_set.image) == expected
    assert isinstance(image_set.image[0], Path)

    assert image_set.dirty is not None
    assert len(image_set.dirty) == expected
    assert isinstance(image_set.dirty[0], Path)

    assert image_set.model is not None
    assert len(image_set.model) == expected
    assert isinstance(image_set.model[0], Path)

    assert image_set.residual is not None
    assert len(image_set.residual) == expected
    assert isinstance(image_set.residual[0], Path)

    assert image_set.psf is not None
    assert len(image_set.psf) == 5  # PSF is the same across all pols
    assert isinstance(image_set.psf[0], Path)

    assert image_set.image[0] == Path("JackSparrow-0000-I-image.fits")
    assert image_set.image[4] == Path("JackSparrow-MFS-I-image.fits")
    assert image_set.image[5] == Path("JackSparrow-0000-Q-image.fits")
    assert image_set.image[9] == Path("JackSparrow-MFS-Q-image.fits")


def test_wsclean_output_named_nomfs():
    image_set = get_wsclean_output_names(
        prefix="JackSparrow", subbands=4, include_mfs=False
    )

    assert isinstance(image_set, ImageSet)
    assert image_set.prefix == "JackSparrow"

    assert image_set.image is not None
    assert len(image_set.image) == 4
    assert isinstance(image_set.image[0], Path)

    assert image_set.dirty is not None
    assert len(image_set.dirty) == 4
    assert isinstance(image_set.dirty[0], Path)

    assert image_set.model is not None
    assert len(image_set.model) == 4
    assert isinstance(image_set.model[0], Path)

    assert image_set.residual is not None
    assert len(image_set.residual) == 4
    assert isinstance(image_set.residual[0], Path)

    assert image_set.psf is not None
    assert len(image_set.psf) == 4
    assert isinstance(image_set.psf[0], Path)
