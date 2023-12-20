"""Basic tests around some of the utility functions
"""

import os
from pathlib import Path

from flint.utils import create_cd_into_directory


def test_create_cd_into_directory(tmpdir):
    example = tmpdir / Path("example_directory")

    assert not example.exists(), f"{example} alreadt exists"

    create_cd_into_directory(in_folder=example)

    assert Path(os.getcwd()) == example
