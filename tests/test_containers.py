"""Items to test functionality around the containers interface"""

from __future__ import annotations

from flint.containers import (
    KNOWN_CONTAINER_LOOKUP,
    LIST_OF_KNOWN_CONTAINERS,
    FlintContainer,
    log_known_containers,
)


def test_all_flint_containers():
    """Make sure everything we know is a FlintContainer"""
    assert all([isinstance(fc, FlintContainer) for fc in LIST_OF_KNOWN_CONTAINERS])


def test_all_known_containers():
    """Same as above but for the known donctainers lookup"""
    for k, v in KNOWN_CONTAINER_LOOKUP.items():
        assert isinstance(k, str)
        assert isinstance(v, FlintContainer)


def test_log_containers():
    """Output all the known containers"""
    # This should simply not error
    log_known_containers()
