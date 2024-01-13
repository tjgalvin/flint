"""Small tests for items related to measurement sets 
and the MS class
"""
from pathlib import Path

from flint.ms import MS


def test_ms_self_attribute():
    ex = Path("example/jack_sparrow.ms")
    ms = MS(path=ex)

    assert ms.ms.path == ex
