"""Some tests related to using aoccalibrate related things
"""
from pathlib import Path
import pytest
import shutil

import pkg_resources
import numpy as np

from flint.calibrate.aocalibrate import AOSolutions, plot_solutions
from flint.bptools.smoother import divide_bandpass_by_ref_ant


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
    assert ao.nchan == 288
    assert ao.npol == 4


def test_aosols_bandpass_plot(ao_sols):
    # This is just a dumb test to make sure the function runs
    plot_solutions(solutions=ao_sols, ref_ant=0)
    plot_solutions(solutions=ao_sols, ref_ant=None)


def test_aosols_bandpass_ref_nu_rank_error(ao_sols):
    ao = AOSolutions.load(path=ao_sols)

    # This should raise an assertion error since the data shape is not right
    with pytest.raises(AssertionError) as ae:
        divide_bandpass_by_ref_ant(complex_gains=ao.bandpass, ref_ant=0)


def test_aosols_bandpass_ref_nu(ao_sols):
    ao = AOSolutions.load(path=ao_sols)

    complex_gains = divide_bandpass_by_ref_ant(complex_gains=ao.bandpass[0], ref_ant=0)

    expected = np.array(
        [
            0.11008759 - 0.00000000e00j,
            0.11009675 - 4.33444224e-19j,
            0.11017988 - 0.00000000e00j,
            0.10990718 - 0.00000000e00j,
            0.11060902 - 8.66905258e-19j,
        ]
    )
    assert np.allclose(expected, complex_gains[0, :5, 0])


# TODO: Need to write more tests for the smoothing and other things
