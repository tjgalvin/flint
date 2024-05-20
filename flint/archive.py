"""Operations around preserving files and products from an flint run"""

import tarfile
from argparse import ArgumentParser
from glob import glob
from pathlib import Path
from typing import NamedTuple, Collection, List, Tuple

from sklearn import base

from flint.logging import logger


class ArchiveOptions(NamedTuple):
    """Container for options related to archiving products from flint workflows"""

    file_globs: Collection[str] = ("*fits", "*png", "*.ms.zip")
    """Glob expressions to use to collect files that should be tarballed"""


def resolve_glob_expressions(
    base_path: Path, glob_expressions: Collection[str]
) -> Tuple[Path, ...]:
    """Collect a set of files given a base directory and a set of glob expressions. Unique
    paths are returned.

    Args:
        base_path (Path): The base folder with files to consider
        glob_expressions (Collection[str]): An iterable with a set of glob expressions to evaluate

    Returns:
        Tuple[Path,...]: Unique collection of paths
    """
    base_path = Path(base_path)

    resolved_files: List[Path] = []

    logger.info(f"Searching {base_path=}")

    for glob_expression in glob_expressions:
        resolved_files.extend(list(base_path.glob(glob_expression)))

    logger.info(
        f"Resolved {len(resolved_files)} files from {len(glob_expressions)} expressions in {base_path=}"
    )

    return tuple(set(resolved_files))


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Operations around archiving")

    subparser = parser.add_subparsers(
        dest="mode", help="Operation mode of flint_archive"
    )

    return parser


def cli() -> None:
    parser = get_parser()

    args = parser.parse_args()


if __name__ == "__main__":
    cli()
