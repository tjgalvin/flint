"""Some tests related to using aoccalibrate related things
"""
from pathlib import Path
import pytest
import shutil

import pkg_resources

from flint.calibrate.aocalibrate import AOSolutions


@pytest.fixture
def ao_sols(tmpdir):
    ao_sols = Path(
        pkg_resources.resource_filename(
            "flint", "data/tests/SB39433.B1934-638.beam0.calibrate.bin"
        )
    )

    out_ao_sols = Path(tmpdir) / ao_sols.name

    shutil.copyfile(ao_sols, out_ao_sols)

    return out_ao_sols


def test_load_aosols(ao_sols):
    ao = AOSolutions.load(path=ao_sols)

    assert ao.nant == 36
