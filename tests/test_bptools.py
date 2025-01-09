"""Itemss around testing components of bptools"""

from __future__ import annotations

import numpy as np
import pytest

from flint.bptools.preflagger import (
    construct_jones_over_max_amp_flags,
    construct_mesh_ant_flags,
)


def count_nan(data):
    return np.sum(~np.isfinite(data))


def test_construct_mesh_ant_flags():
    shape = (1, 36, 288, 4)
    a = np.arange(np.prod(shape)).reshape(shape) * 1.0

    assert count_nan(a) == 0

    a[0, 0, 10:20, :] = np.nan
    assert count_nan(a) == 40

    mask = construct_mesh_ant_flags(mask=~np.isfinite(a[0]))
    assert np.sum(mask) == 1440

    a[0, mask] = np.nan
    assert count_nan(a) == 1440


def test_construct_mesh_ant_flags_assert():
    shape = (1, 36, 288, 4)
    a = np.arange(np.prod(shape)).reshape(shape) * 1.0

    with pytest.raises(AssertionError):
        construct_mesh_ant_flags(mask=~np.isfinite(a))


def test_construct_jobnes_over_max_amp_flags():
    shape = (1, 36, 288, 4)
    a = np.arange(np.prod(shape)).reshape(shape) * 1.0

    a[0, 10, 10, 0] = 100000000
    max_amplitude = 100000
    assert np.sum(a > max_amplitude) == 1

    mask = construct_jones_over_max_amp_flags(
        complex_gains=a, max_amplitude=max_amplitude
    )
    assert np.sum(mask) == 4

    a[mask] = np.nan
    assert count_nan(a) == 4


def test_construct_jobnes_over_max_amp_flags2():
    shape = (1, 36, 288, 4)
    a = np.arange(np.prod(shape)).reshape(shape) * 1.0

    a[0, 10, 10, 0] = 100000000
    a[0, 10, 11, 1] = 100000000
    a[0, 11, 12, 2] = 100000000
    a[0, 12, 13, 3] = 100000000
    max_amplitude = 100000
    assert np.sum(a > max_amplitude) == 4

    mask = construct_jones_over_max_amp_flags(
        complex_gains=a, max_amplitude=max_amplitude
    )
    assert np.sum(mask) == 16

    a[mask] = np.nan
    assert count_nan(a) == 16


def test_construct_jobnes_over_max_amp_flags3():
    shape = (1, 36, 288, 5)
    a = np.arange(np.prod(shape)).reshape(shape) * 1.0

    with pytest.raises(AssertionError):
        construct_jones_over_max_amp_flags(complex_gains=a, max_amplitude=1000000)
