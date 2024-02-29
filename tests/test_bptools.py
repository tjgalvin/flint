"""Itemss around testing components of bptools
"""

import pytest
import numpy as np

from flint.bptools.preflagger import construct_mesh_ant_flags


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
