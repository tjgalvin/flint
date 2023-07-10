"""Utility functions to carry out flagging against ASKAP measurement sets
"""

from typing import NamedTuple
from pathlib import Path
from argparse import ArgumentParser

from flint.logging import logger
from flint.ms import MS, check_column_in_ms, describe_ms
from flint.exceptions import MSError
from flint.sclient import run_singularity_command


class AOFlaggerCommand(NamedTuple):
    """The command to use when running aoflagger"""

    cmd: str
    """The command that will be executed"""
    ms_path: Path
    """The path to the MS that will be flagged. """


def create_aoflagger_cmd(ms: MS) -> AOFlaggerCommand:
    """Create a command to run aoflagger against a measurement set

    Args:
        ms (MS): The measurement set to flag. The column attached to the MS.column
        is flagged.

    Raises:
        MSError: Raised when the attached column is not found in the MS

    Returns:
        AOFlaggerCommand: The aoflagger command that will be run
    """
    logger.info(f"Creating an AOFlagger command. ")

    assert (
        ms.column is not None
    ), f"MS column must be set in order to flag, currently {ms.column=}."

    if not check_column_in_ms(ms):
        raise MSError(f"Column {ms.column} not found in {ms.path}.")

    cmd = f"aoflagger -column {ms.column} -v {str(ms.path.absolute())}"

    return AOFlaggerCommand(cmd=cmd, ms_path=ms.path)


def run_aoflagger_cmd(aoflagger_cmd: AOFlaggerCommand, container: Path) -> None:
    """Run the aoflagger command constructed in its singularity container

    Args:
        aoflagger_cmd (AOFlaggerCommand): The command that will be executed
        container (Path): Path to the container that contains aoflagger
    """
    assert (
        container.exists()
    ), f"The applysolutions container {container} does not exist. "

    bind_dirs = [aoflagger_cmd.ms_path.parent.absolute()]
    logger.debug(f"Bind directory for aoflagger: {bind_dirs}")

    run_singularity_command(
        image=container.absolute(), command=aoflagger_cmd.cmd, bind_dirs=bind_dirs
    )


def flag_ms_aoflagger(ms: MS, container: Path) -> MS:
    """Create and run an aoflagger command in a container

    Args:
        ms (MS): The measurement set with nominated column to flag
        container (Path): The container with the aoflagger program

    Returns:
        MS: Measurement set flagged with the appropriate column
    """
    logger.info(f"Will flag column {ms.column} in {str(ms.path)}.")
    aoflagger_cmd = create_aoflagger_cmd(ms=ms)

    logger.info(f"Flagging command constructed. ")
    run_aoflagger_cmd(aoflagger_cmd=aoflagger_cmd, container=container)

    return ms


def get_parser() -> ArgumentParser:
    """Create the argument parser for the flagging

    Returns:
        ArgumentParser: aoflagger argument parser
    """
    parser = ArgumentParser(description="Run aoflagger against a measurement set")

    parser.add_argument("ms", type=Path, help="The measurement set to flag")
    parser.add_argument(
        "--aoflagger-container",
        type=Path,
        default=Path("aoflagger.sif"),
        help="The container that holds the aoflagger application",
    )
    parser.add_argument(
        "--column", type=str, default="DATA", help="The column of data in MS to flag"
    )

    return parser


def cli() -> None:
    import logging

    logger.setLevel(logging.INFO)

    parser = get_parser()

    args = parser.parse_args()

    ms = MS(path=args.ms, column=args.column)

    describe_ms(ms)
    flag_ms_aoflagger(ms=ms, container=args.aoflagger_container)
    describe_ms(ms)


if __name__ == "__main__":
    cli()
