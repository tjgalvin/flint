"""Small tests for items related to measurement sets
and the MS class
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from flint.calibrate.aocalibrate import ApplySolutions
from flint.exceptions import MSError
from flint.ms import MS, get_phase_dir_from_ms, copy_and_preprocess_casda_askap_ms
from flint.utils import get_packaged_resource_path


@pytest.fixture
def casda_example(tmpdir):
    ms_zip = Path(
        get_packaged_resource_path(
            package="flint.data.tests",
            filename="scienceData.EMU_0529-60.SB50538.EMU_0529-60.beam08_averaged_cal.leakage.ms.zip",
        )
    )
    outpath = Path(tmpdir) / "extract"

    shutil.unpack_archive(ms_zip, outpath)

    ms_path = (
        Path(outpath)
        / "scienceData.EMU_0529-60.SB50538.EMU_0529-60.beam08_averaged_cal.leakage.ms"
    )

    return ms_path


def test_copy_preprocess_ms(casda_example, tmpdir):
    """Run the copying and preprocessing for the casda askap. This is not testing the actual contents or the
    output visibility file yet. Just sanity around the process."""

    output_path = Path(tmpdir) / "casda_ms"

    new_ms = copy_and_preprocess_casda_askap_ms(
        casda_ms=Path(casda_example), output_directory=output_path
    )

    # wjem file format not recgonised
    with pytest.raises(ValueError):
        copy_and_preprocess_casda_askap_ms(
            casda_ms=Path(casda_example) / "Thisdoesnotexist",
            output_directory=output_path,
        )

    # When input directory does not exist
    with pytest.raises(AssertionError):
        copy_and_preprocess_casda_askap_ms(
            casda_ms=Path(
                "thisdoesnotexist/scienceData.EMU_0529-60.SB50538.EMU_0529-60.beam08_averaged_cal.leakage.ms"
            ),
            output_directory=output_path / "New",
        )


@pytest.fixture
def ms_example(tmpdir):
    ms_zip = Path(
        get_packaged_resource_path(
            package="flint.data.tests",
            filename="SB39400.RACS_0635-31.beam0.small.ms.zip",
        )
    )
    outpath = Path(tmpdir) / "39400"

    shutil.unpack_archive(ms_zip, outpath)

    ms_path = Path(outpath) / "SB39400.RACS_0635-31.beam0.small.ms"

    return ms_path


def test_phase_dir(ms_example):
    pos = get_phase_dir_from_ms(ms=ms_example)

    assert np.isclose(pos.ra.deg, 98.211959)
    assert np.isclose(pos.dec.deg, -30.86099889)


def test_ms_self_attribute():
    ex = Path("example/jack_sparrow.ms")
    ms = MS(path=ex)

    assert ms.ms.path == ex


def test_ms_from_options():
    path = Path("example.ms")
    solutions = ApplySolutions(
        cmd="none", solution_path=Path("example_sols.bin"), ms=MS(path=path)
    )

    example_ms = MS.cast(solutions)
    ms = MS(path=path)

    assert isinstance(example_ms, MS)
    assert example_ms == ms


def test_raise_error_ms_from_options():
    path = Path("example.ms")
    solutions = ApplySolutions(
        cmd="none", solution_path=Path("example_sols.bin"), ms=path
    )

    with pytest.raises(MSError):
        MS.cast(solutions)
