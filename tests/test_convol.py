"""Bits around testing the convolution utilities"""

import pytest
import shutil
from pathlib import Path

from flint.convol import get_cube_common_beam
from flint.utils import get_packaged_resource_path


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


def test_get_cube_common_beam(cube_fits) -> None:
    """Ensure that the common beam functionality of from beamcon_3D"""
    fits_files = list(cube_fits.glob("*sub.fits"))
    assert len(fits_files) == 10

    _ = get_cube_common_beam(cube_paths=fits_files, cutoff=150.0)
