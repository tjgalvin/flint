"""Basic tests around some of the utility functions
"""

import os
from pathlib import Path

from flint.utils import create_cd_into_directory, remove_files_folders


def test_create_cd_into_directory(tmpdir):
    example = tmpdir / Path("example_directory")

    assert not example.exists(), f"{example} alreadt exists"

    create_cd_into_directory(in_folder=example)

    assert Path(os.getcwd()) == example


def test_remove_file_folders(tmpdir):
    example_files = [Path(tmpdir) / f for f in ["a", "b", "c", "d"]]
    example_dirs = [Path(tmpdir) / d for d in ["qq", "ww", "ee"]]

    for f in example_files:
        print(f)
        open_file = open(f, "w")
        open_file.write(f"testing {f}")

    for d in example_dirs:
        print(d)
        d.mkdir()

    del_files = remove_files_folders(*example_files[:2], *example_dirs[:2])

    assert len(del_files) == 4
    assert all([not f.exists() for f in example_files[:2]])
    assert example_files[2].exists()
    assert example_dirs[2].exists()
