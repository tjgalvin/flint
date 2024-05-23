"""Tests around archives"""

import tarfile
from pathlib import Path

import pytest

from flint.archive import (
    DEFAULT_TAR_RE_PATTERNS,
    ArchiveOptions,
    copy_files_into,
    create_sbid_tar_archive,
    get_parser,
    resolve_glob_expressions,
    tar_files_into,
)

FILES = [f"some_file_{a:02d}-image.fits" for a in range(36)] + [
    f"a_validation.{ext}" for ext in ("png", "jpeg", "pdf")
]


@pytest.fixture
def glob_files(tmpdir):
    """Create an example set of temporary files in a known directory"""
    for f in FILES:
        touch_file = f"{str(tmpdir / f)}"
        with open(touch_file, "w") as out_file:
            out_file.write("example\n")

    return (tmpdir, FILES)


@pytest.fixture
def temp_files(glob_files):
    base_dir, files = glob_files

    archive_options = ArchiveOptions()

    resolved = resolve_glob_expressions(
        base_path=base_dir, file_re_patterns=archive_options.tar_file_re_patterns
    )

    return (base_dir, resolved)


def test_copy_files_into(tmpdir, temp_files):
    """Basic sanity checks to the copying process"""
    base_dir, files = temp_files

    tmpdir = Path(tmpdir)
    copy_out = tmpdir / "copy_location"

    copy_files_into(copy_out_path=copy_out, files_to_copy=files)

    files_copied = list(copy_out.glob("*"))
    assert len(files_copied) == len(files)
    assert len(files_copied) > 0


def test_glob_expressions(glob_files):
    """Trying out the globbing from the archive options"""
    base_dir, files = glob_files

    archive_options = ArchiveOptions()
    assert len(archive_options.tar_file_re_patterns) > 0

    resolved = resolve_glob_expressions(
        base_path=base_dir, file_re_patterns=archive_options.tar_file_re_patterns
    )

    assert all([isinstance(p, Path) for p in resolved])
    assert len(resolved) == 37


def test_glob_expressions_uniq(glob_files):
    """Make sure that the uniqueness is correct"""
    base_dir, files = glob_files

    archive_options = ArchiveOptions(tar_file_re_patterns=(".*png", ".*png"))
    resolved = resolve_glob_expressions(
        base_path=base_dir, file_re_patterns=archive_options.tar_file_re_patterns
    )
    assert len(resolved) == 1


def test_glob_expressions_empty(glob_files):
    """Make sure that the uniqueness is correct"""
    base_dir, files = glob_files

    archive_options = ArchiveOptions(tar_file_re_patterns=(".*doesnotexist",))
    resolved = resolve_glob_expressions(
        base_path=base_dir, file_re_patterns=archive_options.tar_file_re_patterns
    )
    assert len(resolved) == 0


def test_archive_parser(glob_files):
    """Make sure pirates understand parsers"""
    base_dir, files = glob_files
    parser = get_parser()

    args = parser.parse_args("list".split())

    assert isinstance(args.base_path, Path)
    assert args.file_patterns == DEFAULT_TAR_RE_PATTERNS

    example_path = Path("this/no/exist")
    args = parser.parse_args(f"list --base-path {str(example_path)}".split())
    assert isinstance(args.base_path, Path)
    assert args.base_path == example_path

    example_path = Path(base_dir)
    args = parser.parse_args(
        f"list --base-path {str(example_path)} --file-patterns *pdf".split()
    )
    assert isinstance(args.base_path, Path)
    assert args.base_path == example_path
    assert args.file_patterns == ["*pdf"]

    cmd = r"create --tar-file-patterns '.*linmos.*' '.*MFS.*' '.*beam[0-9]+\.round4-????-image\.fits' --base-path 39420 test_archive_tarball/39420.tar"
    args = parser.parse_args(cmd.split())
    assert len(args.tar_file_patterns) == 3


def test_tar_ball_files(temp_files):
    """Ensure that the tarballing works"""
    base_dir, files = temp_files

    tar_out_path = Path(base_dir) / "some_tarball.tar"
    _ = tar_files_into(tar_out_path=tar_out_path, files_to_tar=files)

    assert tarfile.is_tarfile(tar_out_path)

    with pytest.raises(FileExistsError):
        _ = tar_files_into(tar_out_path=tar_out_path, files_to_tar=files)


def test_create_sbid_archive(glob_files):
    """Attempts to ensure the entire find files and tarball creation works"""
    base_dir, files = glob_files

    archive_options = ArchiveOptions()
    tar_out_path = Path(base_dir) / "example_tarball_2.tar"

    _ = create_sbid_tar_archive(
        base_path=base_dir, tar_out_path=tar_out_path, archive_options=archive_options
    )

    assert tarfile.is_tarfile(tar_out_path)
