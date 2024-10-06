"""Bits around testing the convolution utilities"""

import pytest
import shutil
from pathlib import Path

import numpy as np
from astropy.io import fits

from flint.convol import (
    check_if_cube_fits,
    get_cube_common_beam,
    BeamShape,
)
from flint.utils import get_packaged_resource_path


@pytest.fixture
def image_fits() -> Path:
    image = Path(
        get_packaged_resource_path(
            package="flint.data.tests",
            filename="SB39400.RACS_0635-31.beam0-MFS-subimage_rms.fits",
        )
    )

    return image


@pytest.fixture
def cube_fits(tmpdir) -> Path:
    tmp_dir = Path(tmpdir)
    cube_dir = Path(tmp_dir / "cubes")
    cube_dir.mkdir(parents=True, exist_ok=True)

    cubes_zip = Path(
        get_packaged_resource_path(
            package="flint.data.tests", filename="sub_cube_fits_examples.zip"
        )
    )
    assert cubes_zip.exists()
    shutil.unpack_archive(cubes_zip, cube_dir)

    return cube_dir


def test_check_if_cube_fits(cube_fits, image_fits):
    """See if the cube fits checker is picking up cubes with 3 axis"""
    fits_files = list(cube_fits.glob("*sub.fits"))
    assert len(fits_files) == 10
    assert all([check_if_cube_fits(fits_file=f) for f in fits_files])

    assert not check_if_cube_fits(fits_file=image_fits)
    assert not check_if_cube_fits(fits_file=Path("ThisDoesNotExist"))


def test_get_cube_common_beam_and_convol_cubes(cube_fits) -> None:
    """Ensure that the common beam functionality of from beamcon_3D. Also test the
    convolution to the cubes, as the initial compute can be expensive"""
    fits_files = list(cube_fits.glob("*sub.fits"))
    assert len(fits_files) == 10

    data = fits.getdata(fits_files[0])
    data_shape = np.squeeze(data).shape  # type: ignore

    beam_list = get_cube_common_beam(cube_paths=fits_files, cutoff=150.0)
    assert len(beam_list) == data_shape[0]
    assert all([isinstance(b, BeamShape) for b in beam_list])

    # This appears to make pytest lock up
    from flint.convol import convolve_cubes

    cube_paths = convolve_cubes(
        cube_paths=fits_files,
        beam_shapes=beam_list,
        cutoff=150.0,
        executor_type="process",
    )
    assert all([isinstance(p, Path) for p in cube_paths])
    assert all([p.exists() for p in cube_paths])
