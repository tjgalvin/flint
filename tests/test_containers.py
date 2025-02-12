"""Items to test functionality around the containers interface"""

from __future__ import annotations

from pathlib import Path

from flint.containers import (
    KNOWN_CONTAINER_LOOKUP,
    LIST_OF_KNOWN_CONTAINERS,
    FlintContainer,
    log_known_containers,
    verify_known_containers,
)


def test_verify_known_containers(tmpdir):
    """Check that our verify function works. This will cheat and
    crate some temporary file with the expected file name as the
     verify function in of itself (currently) only checks to see
     if a file exists"""
    container_directory = Path(tmpdir) / "containers1"
    container_directory.mkdir(parents=True)

    assert not verify_known_containers(container_directory=container_directory)

    for cata in LIST_OF_KNOWN_CONTAINERS:
        cata_path = container_directory / cata.file_name
        cata_path.touch()

    assert verify_known_containers(container_directory=container_directory)


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
