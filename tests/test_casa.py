"""Tests around the casa self-calibration tooling"""

from __future__ import annotations

from flint.selfcal.casa import args_to_casa_task_string


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
