"""Operations around preserving files and products from an flint run"""

import tarfile
from argparse import ArgumentParser
from glob import glob
from pathlib import Path
from typing import NamedTuple, Collection, List, Tuple

from sklearn import base

from flint.logging import logger

DEFAULT_GLOB_EXPRESSIONS = (
    "*image*fits",
    "*linmos*",
    "*yaml",
    "*.txt",
    "*png",
    "*.ms.zip",
)


class ArchiveOptions(NamedTuple):
    """Container for options related to archiving products from flint workflows"""

    file_globs: Collection[str] = DEFAULT_GLOB_EXPRESSIONS
    """Glob expressions to use to collect files that should be tarballed"""


def resolve_glob_expressions(
    base_path: Path, file_globs: Collection[str]
) -> Tuple[Path, ...]:
    """Collect a set of files given a base directory and a set of glob expressions. Unique
    paths are returned.

    Args:
        base_path (Path): The base folder with files to consider
        filke_globs (Collection[str]): An iterable with a set of glob expressions to evaluate

    Returns:
        Tuple[Path,...]: Unique collection of paths
    """
    base_path = Path(base_path)

    resolved_files: List[Path] = []

    logger.info(f"Searching {base_path=}")

    for glob_expression in file_globs:
        resolved_files.extend(list(base_path.glob(glob_expression)))

    logger.info(
        f"Resolved {len(resolved_files)} files from {len(file_globs)} expressions in {base_path=}"
    )

    return tuple(set(resolved_files))


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
        base_path=base_path, file_globs=archive_options.file_globs
    )

    tar_out_path = tar_files_into(tar_out_path=tar_out_path, files_to_tar=files_to_tar)

    return tar_out_path


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Operations around archiving")

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
        "--file-globs",
        nargs="+",
        default=DEFAULT_GLOB_EXPRESSIONS,
        type=str,
        help="The glob expressions to evaluate",
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
        "--file-globs",
        nargs="+",
        default=DEFAULT_GLOB_EXPRESSIONS,
        type=str,
        help="The glob expressions to evaluate inside the base path directory",
    )

    return parser


def cli() -> None:
    parser = get_parser()

    args = parser.parse_args()

    if args.mode == "list":
        archive_options = ArchiveOptions(file_globs=args.file_globs)

        files = resolve_glob_expressions(
            base_path=args.base_path, file_globs=archive_options.file_globs
        )

        for file in sorted(files):
            logger.info(f"{file}")
    elif args.mode == "create":
        archive_options = ArchiveOptions(file_globs=args.file_globs)

        create_sbid_tar_archive(
            tar_out_path=args.tar_out_path,
            base_path=args.base_path,
            archive_options=archive_options,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    cli()
