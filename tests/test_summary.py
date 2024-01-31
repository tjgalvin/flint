import pytest
import shutil
from pathlib import Path

import pkg_resources
from astropy.time import Time
from astropy.coordinates import EarthLocation

from flint.ms import get_times_from_ms, get_telescope_location_from_ms
from flint.summary import (
    FieldSummary,
    create_field_summary,
    add_rms_information,
    update_field_summary,
)
from flint.source_finding.aegean import AegeanOutputs


@pytest.fixture
def aegean_outputs_example():
    rms = Path(
        pkg_resources.resource_filename(
            "flint", "data/tests/SB39400.RACS_0635-31.beam0-MFS-subimage_rms.fits"
        )
    )

    comp = Path(
        pkg_resources.resource_filename(
            "flint", "data/tests/SB38959.RACS_1357-18.noselfcal.linmos_comp.fits"
        )
    )

    aegean_outputs = AegeanOutputs(bkg=rms, rms=rms, comp=comp)

    return aegean_outputs


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


def test_field_summary(ms_example):
    sbid_path = ms_example
    cal_sbid_path = Path("/scratch3/gal16b/split/39433/SB39433.1934-638.beam0.ms")

    field_summary = create_field_summary(ms=sbid_path, cal_sbid_path=cal_sbid_path)

    assert isinstance(field_summary, FieldSummary)
    assert field_summary.sbid == "39400"
    assert field_summary.cal_sbid == "39433"

    field_summary = create_field_summary(ms=sbid_path)

    assert isinstance(field_summary, FieldSummary)
    assert field_summary.sbid == "39400"
    assert field_summary.cal_sbid is None
    assert isinstance(field_summary.location, EarthLocation)
    assert field_summary.integration_time == 19.90655994415036
    assert isinstance(field_summary.ms_times, Time)
    assert len(field_summary.ms_times) == 2


def test_field_summary_rms_info(ms_example, aegean_outputs_example):
    sbid_path = ms_example
    cal_sbid_path = Path("/scratch3/gal16b/split/39433/SB39433.1934-638.beam0.ms")

    field_summary = create_field_summary(ms=sbid_path, cal_sbid_path=cal_sbid_path)

    assert isinstance(field_summary, FieldSummary)
    assert field_summary.sbid == "39400"
    assert field_summary.cal_sbid == "39433"
    assert isinstance(field_summary.location, EarthLocation)
    assert field_summary.integration_time == 19.90655994415036
    assert isinstance(field_summary.ms_times, Time)

    field_summary = add_rms_information(
        field_summary=field_summary, aegean_outputs=aegean_outputs_example
    )
    assert field_summary.no_components == 7225


def test_field_summary_rms_info_creation(ms_example, aegean_outputs_example):
    sbid_path = ms_example
    cal_sbid_path = Path("/scratch3/gal16b/split/39433/SB39433.1934-638.beam0.ms")

    field_summary = create_field_summary(
        ms=sbid_path, cal_sbid_path=cal_sbid_path, aegean_outputs=aegean_outputs_example
    )

    assert isinstance(field_summary, FieldSummary)
    assert field_summary.sbid == "39400"
    assert field_summary.cal_sbid == "39433"
    assert isinstance(field_summary.location, EarthLocation)
    assert field_summary.integration_time == 19.90655994415036
    assert isinstance(field_summary.ms_times, Time)


def test_field_summary_update(ms_example, aegean_outputs_example):
    sbid_path = ms_example
    cal_sbid_path = Path("/scratch3/gal16b/split/39433/SB39433.1934-638.beam0.ms")

    field_summary = create_field_summary(ms=sbid_path, cal_sbid_path=cal_sbid_path)

    assert isinstance(field_summary, FieldSummary)
    assert field_summary.sbid == "39400"
    assert field_summary.cal_sbid == "39433"
    assert isinstance(field_summary.location, EarthLocation)
    assert field_summary.integration_time == 19.90655994415036
    assert isinstance(field_summary.ms_times, Time)
    assert field_summary.round is None

    field_summary = update_field_summary(
        field_summary=field_summary, aegean_outputs=aegean_outputs_example, round=2
    )
    assert field_summary.no_components == 7225
    assert field_summary.round == 2


def test_ms_example_times(ms_example):
    times = get_times_from_ms(ms=ms_example)

    assert isinstance(times, Time)
    assert len(times) == 1998


def test_ms_example_telescope(ms_example):
    telescope = get_telescope_location_from_ms(ms=ms_example)

    assert isinstance(telescope, EarthLocation)
