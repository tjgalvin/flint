"""Small tests for items related to measurement sets
and the MS class
"""
import pytest
from pathlib import Path

from flint.exceptions import MSError
from flint.ms import MS
from flint.calibrate.aocalibrate import ApplySolutions


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
