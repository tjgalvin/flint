"""Operations around preserving files and products from an flint run"""

from __future__ import (  # Used for mypy/pylance to like the return type of MS.with_options
    annotations,
)

import re
import shutil
import tarfile
from argparse import ArgumentParser
from pathlib import Path
from typing import Collection, List, NamedTuple, Tuple

from flint.logging import logger

# TODO: Perhaps move these to flint.naming, and can be built up
# based on rules, e.g. imager used, source finder etc.
DEFAULT_TAR_RE_PATTERNS = (
    r".*MFS.*image.*fits",
    r".*linmos.*",
    r".*yaml",
    r".*\.txt",
    r".*png",
    r".*\.ms\.zip",
    r".*\.caltable",
)
DEFAULT_COPY_RE_PATTERNS = (r".*linmos.*fits", r".*png")


class ArchiveOptions(NamedTuple):
    """Container for options related to archiving products from flint workflows"""

    tar_file_re_patterns: Collection[str] = DEFAULT_TAR_RE_PATTERNS
    """Regular-expressions to use to collect files that should be tarballed"""
    copy_file_re_patterns: Collection[str] = DEFAULT_COPY_RE_PATTERNS
    """Regular-expressions used to identify files to copy into a final location (not tarred)"""

    def with_options(self, **kwargs) -> ArchiveOptions:
        opts = self._asdict()
        opts.update(**kwargs)

        return ArchiveOptions(**opts)


def resolve_glob_expressions(
    base_path: Path, file_re_patterns: Collection[str]
) -> Tuple[Path, ...]:
    """Collect a set of files given a base directory and a set of glob expressions. Unique
    paths are returned.

    Args:
        base_path (Path): The base folder with files to consider
        file_re_patterns (Collection[str]): An iterable with a set of regular-expression patterns to evaluate

    Returns:
        Tuple[Path,...]: Unique collection of paths
    """
    base_path = Path(base_path)

    resolved_files: List[Path] = []

    logger.info(f"Searching {base_path=}")

    all_files = list(base_path.iterdir())
    logger.info(f"{len(all_files)} total files and {len(file_re_patterns)} to consider")

    for reg_expression in file_re_patterns:
        logger.info(f"Using expression: {reg_expression}")
        resolved_files.extend(
            [f for f in all_files if re.search(reg_expression, str(f.name))]
        )

    logger.info(
        f"Resolved {len(resolved_files)} files from {len(file_re_patterns)} expressions in {base_path=}"
    )

    return tuple(sorted([Path(p) for p in set(resolved_files)]))


def copy_files_into(copy_out_path: Path, files_to_copy: Collection[Path]) -> Path:
    """Copy a set of specified files into an output directory

    Args:
        copy_out_path (Path): Path to copy files into
        files_to_copy (Collection[Path]): Files that shall be copied

    Returns:
        Path: The path files were copied into
    """

    copy_out_path = Path(copy_out_path)

    copy_out_path.mkdir(parents=True, exist_ok=True)
    total = len(files_to_copy)
    not_copied: List[Path] = []

    logger.info(f"Copying {total} files into {copy_out_path}")
    for count, file in enumerate(files_to_copy):
        logger.info(f"{count+1} of {total}, copying {file}")

        if not file.is_file():
            # TODO: Support folders
            not_copied.append(file)
            logger.critical(f"{file} is not a file. Skipping. ")
            continue

        shutil.copy(file, copy_out_path)

    if not_copied:
        logger.critical(f"Did not copy {len(not_copied)} files, {not_copied=}")

    return copy_out_path


# TODO: Add a clobber option
def tar_files_into(tar_out_path: Path, files_to_tar: Collection[Path]) -> Path:
    """Create a tar file given a desired output path and list of files to tar.

    Args:
        tar_out_path (Path): The output path of the tarball. The parent directory will be created if necessary.
        files_to_tar (Collection[Path]): All the files to tarball up

    Raises:
        FileExistsError: The path of the tarball created

    Returns:
        Path: There exists a tarball of the same name
    """

    tar_out_path = Path(tar_out_path)

    if tar_out_path.exists():
        raise FileExistsError(f"{tar_out_path} already exists. ")

    # Create the output directory in case it does not exist
    tar_out_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(files_to_tar)
    logger.info(f"Taring {total} files")
    logger.info(f"Opening {tar_out_path}")
    with tarfile.open(tar_out_path, "w") as tar:
        for count, file in enumerate(files_to_tar):
            logger.info(f"{count+1} of {total}, adding {str(file)}")
            tar.add(file, arcname=file.name)

    logger.info(f"Created {tar_out_path}")
    return tar_out_path


def create_sbid_tar_archive(
    tar_out_path: Path, base_path: Path, archive_options: ArchiveOptions
) -> Path:
    """Create a tar file of key products in a SBID folder.

    Args:
        tar_out_path (Path): The output location of the tarball to write
        base_path (Path): The base directory that contains files to archive
        archive_options (ArchiveOptions): Options relating to how files are found and archived

    Returns:
        Path: Output tarball directory
    """

    files_to_tar = resolve_glob_expressions(
        base_path=base_path, file_re_patterns=archive_options.tar_file_re_patterns
    )

    tar_out_path = tar_files_into(tar_out_path=tar_out_path, files_to_tar=files_to_tar)

    return tar_out_path


def copy_sbid_files_archive(
    copy_out_path: Path, base_path: Path, archive_options: ArchiveOptions
) -> Path:
    """Copy files from an SBID processing folder into a final location. Uses the
    `copy_file_globs` set of expressions to identify files to copy.

    Args:
        copy_out_path (Path): The output location of the tarball to write
        base_path (Path): The base directory that contains files to archive
        archive_options (ArchiveOptions): Options relating to how files are found and archived

    Returns:
        Path: Output tarball directory
    """

    files_to_copy = resolve_glob_expressions(
        base_path=base_path, file_re_patterns=archive_options.copy_file_re_patterns
    )

    copy_out_path = copy_files_into(
        copy_out_path=copy_out_path, files_to_copy=files_to_copy
    )

    return copy_out_path


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Operations around archiving. Patterns are specified as regular expressions (not globs). "
    )

    subparser = parser.add_subparsers(
        dest="mode", help="Operation mode of flint_archive"
    )

    list_parser = subparser.add_parser("list")
    list_parser.add_argument(
        "--base-path",
        type=Path,
        default=Path("."),
        help="Base directory to perform glob expressions",
    )

    list_parser.add_argument(
        "--file-patterns",
        nargs="+",
        default=DEFAULT_TAR_RE_PATTERNS,
        type=str,
        help="The regular expression patterns to evaluate",
    )

    create_parser = subparser.add_parser("create", help="Create a tarfile archive")
    create_parser.add_argument(
        "tar_out_path", type=Path, help="Path of the output tar file to be created"
    )
    create_parser.add_argument(
        "--base-path",
        type=Path,
        default=Path("."),
        help="Base directory to perform glob expressions",
    )

    create_parser.add_argument(
        "--tar-file-patterns",
        nargs="+",
        default=DEFAULT_TAR_RE_PATTERNS,
        type=str,
        help="The regular expression patterns to evaluate inside the base path directory",
    )

    create_parser = subparser.add_parser(
        "copy", help="Copy a set of files into a output directory"
    )
    create_parser.add_argument(
        "copy_out_path",
        type=Path,
        help="Path of the output folder that files will be copied into",
    )
    create_parser.add_argument(
        "--base-path",
        type=Path,
        default=Path("."),
        help="Base directory to perform glob expressions",
    )

    create_parser.add_argument(
        "--copy-file-patterns",
        nargs="+",
        default=DEFAULT_COPY_RE_PATTERNS,
        type=str,
        help="The regular expression patterns to evaluate inside the base path directory",
    )

    return parser


def cli() -> None:
    parser = get_parser()

    args = parser.parse_args()

    if args.mode == "list":
        archive_options = ArchiveOptions(tar_file_re_patterns=args.file_patterns)

        files = resolve_glob_expressions(
            base_path=args.base_path,
            file_re_patterns=archive_options.tar_file_re_patterns,
        )

        for count, file in enumerate(sorted(files)):
            logger.info(f"{count} of {len(files)}, {file}")
    elif args.mode == "create":
        archive_options = ArchiveOptions(tar_file_re_patterns=args.tar_file_patterns)

        create_sbid_tar_archive(
            tar_out_path=args.tar_out_path,
            base_path=args.base_path,
            archive_options=archive_options,
        )
    elif args.mode == "copy":
        archive_options = ArchiveOptions(copy_file_re_patterns=args.copy_file_globs)

        copy_sbid_files_archive(
            copy_out_path=args.copy_out_path,
            base_path=args.base_path,
            archive_options=archive_options,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    cli()
