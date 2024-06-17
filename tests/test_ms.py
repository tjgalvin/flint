"""Small tests for items related to measurement sets
and the MS class
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from flint.calibrate.aocalibrate import ApplySolutions
from flint.exceptions import MSError
from flint.ms import (
    MS,
    copy_and_preprocess_casda_askap_ms,
    find_mss,
    get_phase_dir_from_ms,
)
from flint.utils import get_packaged_resource_path


def test_find_mss(tmpdir):
    tmpdir = Path(tmpdir)
    for name in range(45):
        new_ms = tmpdir / f"SB1234.Pirate_1234+456.beam{name}.ms"
        new_ms.mkdir()

        new_folder = tmpdir / f"not_and_ms_{name}.folder"
        new_folder.mkdir()

    res = find_mss(mss_parent_path=tmpdir, expected_ms_count=45)
    assert len(res) == 45

    with pytest.raises(AssertionError):
        _ = find_mss(mss_parent_path=tmpdir, expected_ms_count=49005)


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


def _test_the_data(ms):
    """Some very simple tests for the rotation. The expected numbers come from manually
    stabbing the MSs"""
    from casacore.tables import table

    with table(str(ms), ack=False) as tab:
        data = tab.getcol("DATA")
        inst_data = tab.getcol("INSTRUMENT_DATA")
        colnames = tab.colnames()

    assert all([col in colnames for col in ("DATA", "INSTRUMENT_DATA")])

    expected_inst_data = np.array(
        [5.131794 - 23.130766j, 45.26275 - 45.140232j, 0.80312335 + 0.41873842j],
        dtype=np.complex64,
    )
    assert np.allclose(inst_data[:, 10, 0], expected_inst_data)

    expected_data = np.array(
        [-12.364758 - 59.172283j, -10.334289 - 97.017624j, 1.022179 - 0.18529199j],
        dtype=np.complex64,
    )
    assert np.allclose(data[:, 10, 0], expected_data)


def test_copy_preprocess_ms(casda_example, tmpdir):
    """Run the copying and preprocessing for the casda askap. This is not testing the actual contents or the
    output visibility file yet. Just sanity around the process."""

    output_path = Path(tmpdir) / "casda_ms"

    new_ms = copy_and_preprocess_casda_askap_ms(
        casda_ms=Path(casda_example), output_directory=output_path
    )
    _test_the_data(ms=new_ms.path)

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
