"""Tests around the casa self-calibration tooling"""

from __future__ import annotations

from pathlib import Path

from flint.ms import MS
from flint.selfcal.casa import args_to_casa_task_string, create_and_check_caltable_path


def test_create_solution_path():
    """Create the path to a solutions file, including appropriate handling of a channel range"""

    ms_path = Path("/jack/has/a/measurement_set.ms")
    cal_table = create_and_check_caltable_path(ms=MS.cast(ms_path))

    assert isinstance(cal_table, Path)
    assert cal_table == Path("/jack/has/a/measurement_set.caltable")

    cal_table = create_and_check_caltable_path(
        ms=MS.cast(ms_path), channel_range=(0, 143)
    )

    assert isinstance(cal_table, Path)
    assert cal_table == Path("/jack/has/a/measurement_set.caltable.ch0000-0143")


def test_args_to_casa_task_str():
    """Ensure we can transform casa style function calls to strings"""

    transform = args_to_casa_task_string(
        task="mstransform",
        regridms=True,
        nspw=1,
        mode="channel",
        nchan=-1,
    )
    assert isinstance(transform, str)

    expected = "casa -c mstransform(regridms=True,nspw=1,mode='channel',nchan=-1)"
    assert transform == expected


def test_args_to_casa_task_str_arg_list():
    """Same as above but should an argument be a list of paths (for example) needs help"""

    paths = [
        Path("/jack/dataset1.ms"),
        Path("/jack/dataset2.ms"),
    ]
    applycal = args_to_casa_task_string(
        task="applycal",
        vis=Path("/some/other/ship/visibility.ms"),
        gaintable=paths,
    )

    expected = "casa -c applycal(vis='/some/other/ship/visibility.ms',gaintable=('/jack/dataset1.ms','/jack/dataset2.ms'))"
    assert expected == applycal
