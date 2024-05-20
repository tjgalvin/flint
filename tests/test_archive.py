"""Tests around archives"""

import pytest
from pathlib import Path

from flint.archive import ArchiveOptions, resolve_glob_expressions

FILES = [f"some_file_{a:02d}-image.fits" for a in range(36)] + [
    f"a_validation.{ext}" for ext in ("png", "jpeg", "pdf")
]


@pytest.fixture
def glob_files(tmpdir):

    for f in FILES:
        touch_file = f"{str(tmpdir / f)}"
        with open(touch_file, "w") as out_file:
            out_file.write("example\n")

    return (tmpdir, FILES)


def test_glob_expressions(glob_files):
    """Trying out the globbing from the archive options"""
    base_dir, files = glob_files

    archive_options = ArchiveOptions()
    assert len(archive_options.file_globs) > 0

    resolved = resolve_glob_expressions(
        base_path=base_dir, glob_expressions=archive_options.file_globs
    )

    assert all([isinstance(p, Path) for p in resolved])
    assert len(resolved) == 37


def test_glob_expressions_uniq(glob_files):
    """Make sure that the uniqueness is correct"""
    base_dir, files = glob_files

    archive_options = ArchiveOptions(file_globs=("*png", "*png"))
    resolved = resolve_glob_expressions(
        base_path=base_dir, glob_expressions=archive_options.file_globs
    )
    assert len(resolved) == 1


def test_glob_expressions_empty(glob_files):
    """Make sure that the uniqueness is correct"""
    base_dir, files = glob_files

    archive_options = ArchiveOptions(file_globs=("*doesnotexist",))
    resolved = resolve_glob_expressions(
        base_path=base_dir, glob_expressions=archive_options.file_globs
    )
    assert len(resolved) == 0
