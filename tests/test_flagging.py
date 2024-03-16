"""Test utilities related to flagging measurement set operations
"""

import shutil
from pathlib import Path

import numpy as np
import pytest
from casacore.tables import table

from flint.flagging import flag_ms_zero_uvws
from flint.utils import get_packaged_resource_path


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


def test_flag_ms_zero_uvws(ms_example):
    flag_ms_zero_uvws(ms=ms_example)

    with table(str(ms_example), ack=False) as tab:
        uvws = tab.getcol("UVW")
        flags = tab.getcol("FLAG")

        uvw_mask = np.all(uvws == 0, axis=1)

        assert np.all(flags[uvw_mask] == True)  # noQA: E712


def test_add_flag_ms_zero_uvws(ms_example):
    with table(str(ms_example), readonly=False, ack=False) as tab:
        uvws = tab.getcol("UVW")
        uvws[:1, :] = 0

        tab.putcol("UVW", uvws)

    flag_ms_zero_uvws(ms=ms_example)

    with table(str(ms_example), ack=False) as tab:
        uvws = tab.getcol("UVW")

        flags = tab.getcol("FLAG")

        uvw_mask = np.all(uvws == 0, axis=1)

        assert np.sum(uvw_mask) > 0
        assert np.all(flags[uvw_mask] == True)  # noQA: E712
