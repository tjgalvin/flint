"""Small tests for items related to measurement sets
and the MS class
"""
import shutil
from pathlib import Path

import pkg_resources
import pytest
import numpy as np

from flint.calibrate.aocalibrate import ApplySolutions
from flint.exceptions import MSError
from flint.ms import MS, get_phase_dir_from_ms


@pytest.fixture
def ms_example(tmpdir):
    ms_zip = Path(
        pkg_resources.resource_filename(
            "flint", "data/tests/SB39400.RACS_0635-31.beam0.small.ms.zip"
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
