"""Operations around preserving files and products from an flint run"""

import tarfile
from argparse import ArgumentParser
from glob import glob
from pathlib import Path
from typing import NamedTuple, Collection, List, Tuple

from sklearn import base

from flint.logging import logger

DEFAULT_GLOB_EXPRESSIONS = ("*fits", "*png", "*.ms.zip")


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


if __name__ == "__main__":
    cli()
