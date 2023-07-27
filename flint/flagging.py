"""Utility functions to carry out flagging against ASKAP measurement sets
"""

from typing import NamedTuple, Union, Optional
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
from casacore.tables import table

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


def nan_zero_extreme_flag_ms(
    ms: Union[Path, MS],
    data_column: Optional[str] = None,
    flag_extreme_dxy: bool = True,
    dxy_thresh: float = 4.0,
    nan_data_on_flag: bool = False
) -> MS:
    """Will flag a MS based on NaNs or zeros in the nominated data column of a measurement set.
    These NaNs might be introduced into a column via the application of a applysolutions task.
    Zeros might be introduced by the correlator dropping cycles and not appropriately settting the
    corresponding FLAG column (although this might have been fixed).

    There is also an optional component to flag based on extreme Stokes-V values.

    Visibilities that are marked as bad will have the FLAG column updatede appropriately.

    Args:
        ms (Union[Path,MS]): The measurement set that will be processed and have visibilities flagged.
        data_column (Optional[str], optional): The column to inspect. This will override the value in the nominated column of the MS. Defaults to None.
        flag_extreme_dxy (bool, optional): Whether Stokes-V will be inspected and flagged. Defaults to True.
        dxy_thresh (float, optional): Threshold used in the Stokes-V case. Defaults to 4..
        nan_data_on_flag (bool, optional): If True, data whose FLAG is set to True will become NaNs. Defaults to False. 

    Returns:
        MS: The container of the processed MS
    """
    ms = MS.cast(ms)

    if data_column is None and ms.column is None:
        logger.warn(f"No valid data column selected, using default of DATA")
        data_column = "DATA"
    elif data_column is None and ms.column is not None:
        logger.info(f"Using nominated {ms.column} column for {str(ms.path)}")
        data_column = ms.column

    logger.info(f"Flagging NaNs and zeros in {data_column}.")

    with table(str(ms.path), readonly=False, ack=False) as tab:
        data = tab.getcol(data_column)
        flags = tab.getcol("FLAG")
        
        nan_mask = np.where(~np.isfinite(data))
        zero_mask = np.where(data == 0 + 0j)
        uvw_mask = np.any(tab.getcol("UVW") == 0, axis=1)
        logger.info(
            f"Will flag {np.sum(nan_mask)} NaNs, zero'd data {np.sum(zero_mask)}, zero'd UVW {np.sum(uvw_mask)}. "
        )

        no_flags_before = np.sum(flags)
        # TODO: Consider batching this to allow larger MS being used
        flags[nan_mask] = True
        flags[zero_mask] = True
        flags[uvw_mask] = True

        if flag_extreme_dxy:
            logger.info(f"Flagging based on extreme Stokes-V, threshold {dxy_thresh=}")
            dxy = np.abs(data[:, :, 1] - data[:, :, 2])
            dxy_mask = np.where(dxy > dxy_thresh)
            logger.info(f"Will flag {np.sum(dxy_mask)} extreme Stokes-V.")

            # TODO: This can be compressed to a one-liner
            flags[:, :, 0][dxy_mask] = True
            flags[:, :, 1][dxy_mask] = True
            flags[:, :, 2][dxy_mask] = True
            flags[:, :, 3][dxy_mask] = True

        no_flags_after = np.sum(flags)
        logger.info(
            f"Flags before: {no_flags_before}, Flags after: {no_flags_after}, Difference {no_flags_after - no_flags_before}"
        )

        logger.info(f"Adding updated flags column")
        tab.putcol("FLAG", flags)

        if nan_data_on_flag:
            data[flags==True] = np.nan 
            logger.info(f"Setting {np.sum(flags)} DATA items to NaN.")
            tab.putcol(data_column, data)

    return ms


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
    ), f"MS column must be set in order to flag, currently {ms.column=}. Full {ms=}"

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


def flag_ms_aoflagger(ms: MS, container: Path, rounds: int = 1) -> MS:
    """Create and run an aoflagger command in a container

    Args:
        ms (MS): The measurement set with nominated column to flag
        container (Path): The container with the aoflagger program
        rounds (int, optional): Number of times to run the flagging. Defaults to 1.

    Returns:
        MS: Measurement set flagged with the appropriate column
    """
    logger.info(f"Will flag column {ms.column} in {str(ms.path)}.")
    aoflagger_cmd = create_aoflagger_cmd(ms=ms)

    for i in range(rounds):
        logger.info(f"Flagging command constructed. ")
        run_aoflagger_cmd(aoflagger_cmd=aoflagger_cmd, container=container)

    return ms


def get_parser() -> ArgumentParser:
    """Create the argument parser for the flagging

    Returns:
        ArgumentParser: aoflagger argument parser
    """
    parser = ArgumentParser(description="Run aoflagger against a measurement set")

    subparser = parser.add_subparsers(dest="mode")

    aoflagger_parser = subparser.add_parser(
        "aoflagger", description="Run AOFlagger against an measurement set. "
    )
    aoflagger_parser.add_argument("ms", type=Path, help="The measurement set to flag")
    aoflagger_parser.add_argument(
        "--aoflagger-container",
        type=Path,
        default=Path("aoflagger.sif"),
        help="The container that holds the aoflagger application",
    )
    aoflagger_parser.add_argument(
        "--column", type=str, default="DATA", help="The column of data in MS to flag"
    )

    nan_zero_parser = subparser.add_parser(
        "nanflag", help="Flag visibilities tthat are either NaN or zeros. "
    )
    nan_zero_parser.add_argument(
        "ms", type=Path, help="The measurement set that will be flagged. "
    )
    nan_zero_parser.add_argument(
        "--column", type=str, default="DATA", help="The column of data in MS to flag"
    )
    nan_zero_parser.add_argument(
        "--flag-extreme-dxy",
        action="store_true",
        help="Flag visibilities whose ABS(XY - YX) change significantly",
    )
    nan_zero_parser.add_argument(
        "--dxy-thresh",
        type=float,
        default=4.0,
        help="If extreme dxy flagging, ABS(XY - YX) above this value will be flagged. ",
    )
    nan_zero_parser.add_argument(
        '--nan-data-on-flag',
        action='store_true',
        help='NaN the data if their FLAG attribute is True. '
    )
    return parser


def cli() -> None:
    import logging

    logger.setLevel(logging.INFO)

    parser = get_parser()

    args = parser.parse_args()

    if args.mode == "aoflagger":
        ms = MS(path=args.ms, column=args.column)

        describe_ms(ms)
        flag_ms_aoflagger(ms=ms, container=args.aoflagger_container)
        describe_ms(ms)
    elif args.mode == "nanflag":
        nan_zero_extreme_flag_ms(
            ms=args.ms,
            data_column=args.column,
            flag_extreme_dxy=args.flag_extreme_dxy,
            dxy_thresh=args.dxy_thresh,
            nan_data_on_flag=args.nan_data_on_flag
        )


if __name__ == "__main__":
    cli()
