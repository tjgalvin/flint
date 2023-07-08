"""Code to use AO calibrate s
"""
from __future__ import annotations  # used to keep mypy/pylance happy in AOSolutions
from pathlib import Path
from typing import NamedTuple, Optional
from argparse import ArgumentParser

import numpy as np
from spython.main import Client as sclient

from flint.logging import logger
from flint.ms import MS
from flint.sclient import run_singularity_command


class CalibrateCommand(NamedTuple):
    """The AO Calibrate command and output path of the corresponding solutions file"""

    cmd: str
    """The calibrate command that will be executed
    """
    solution_path: Path
    """The output path of the solutions file
    """
    ms: MS


class ApplySolutionsCommand(NamedTuple):
    """The applysolutions command to execute"""

    cmd: str
    """The command that will be executed"""
    solution_path: Path
    """Location of the solutions file to apply"""
    ms: MS
    """The measurement set that will have the solutions applied to"""


# TODO: Rename the bandpass attribute?
class AOSolutions(NamedTuple):
    """Structure to load an AO-style solutions file"""

    path: Path
    """Path of the solutions file loaded"""
    nsol: int
    """Number of time solutions"""
    nant: int
    """Number of antenna in the solution file"""
    nchan: int
    """Number of channels in the solution file"""
    npol: int
    """Number of polarisations in the file"""
    bandpass: np.ndarray
    """Complex data representing the antennea Jones. Shape is (nsol, nant, nchan, npol)"""

    @classmethod
    def load(cls, path: Path) -> AOSolutions:
        """Load in an AO-stule solution file. See `load_solutions_file`, which is
        internally used.
        """
        return load_aosolutions_file(solutions_path=path)


CALIBRATE_SUFFIX = ".calibrate.bin"


def load_aosolutions_file(solutions_path: Path) -> AOSolutions:
    """Load in an AO-style solutions file

    Args:
        solutions_path (Path): The path of the solutions file to load

    Returns:
        AOSolutions: Structure container the deserialized solutions file
    """

    assert (
        solutions_path.exists() and solutions_path.is_file()
    ), f"{str(solutions_path)} either does not exist or is not a file. "
    logger.info(f"Loading {solutions_path}")

    with open(solutions_path, "r") as in_file:
        _junk = np.fromfile(in_file, dtype="<i4", count=2)

        header = np.fromfile(in_file, dtype="<i4", count=10)
        logger.info(f"Header extracted: {header=}")
        file_type = header[0]
        assert file_type == 0, f"Expected file_type of 0, found {file_type}"

        structure_type = header[1]
        assert file_type == 0, f"Expected structure_type of 0, found {structure_type}"

        nsol, nant, nchan, npol = header[2:6]
        sol_shape = (nsol, nant, nchan, npol)

        # TODO: Ask Emil why there is the sqrt(2) here
        bandpass = np.sqrt(2) / np.fromfile(
            in_file, dtype="<c16", count=np.prod(sol_shape)
        ).reshape(sol_shape)
        logger.info(f"Loaded solutions of shape {bandpass.shape}")

        return AOSolutions(
            path=solutions_path,
            nsol=nsol,
            nant=nant,
            nchan=nchan,
            npol=npol,
            bandpass=bandpass,
        )


def create_calibrate_cmd(
    ms: MS, calibrate_model: Path, solution_path: Optional[Path] = None, **kwargs
) -> CalibrateCommand:
    """Generate a typical ao calibrate command. Any extra keyword arguments
    are passed through as additional options to the `calibrate` program.

    Args:
        ms (MS): The measurement set to calibrate. There needs to be a nominated data_column.
        calibrate_model (Path): Path to a generated calibrate sky-model
        solution_path (Path, optional): The output path of the calibrate solutions file.
        If None, a default suffix of "calibrate.bin" is used. Defaults to None.

    Raises:
        FileNotFoundError: Raised when calibrate_model can not be found.

    Returns:
        CalibrateCommand: The calibrate command to execute and output solution file
    """
    logger.info(f"Creating calibrate command for {ms.path}")

    # This is a typical calibrate command.
    # calibrate -minuv 100 -i 50 -datacolumn DATA
    #        -m 2022-04-14_100122_0.calibrate.txt
    #        2022-04-14_100122_0.ms 2022-04-14_100122_0.aocalibrate.bin

    assert ms.column is not None, f"{ms} does not have a nominated data_column"

    if not calibrate_model.exists():
        raise FileNotFoundError(f"Calibrate model {calibrate_model} not found. ")

    solution_path = (
        ms.path.with_suffix(CALIBRATE_SUFFIX)
        if solution_path is None
        else solution_path
    )

    calibrate_kwargs: str = " ".join([f"-{key} {item}" for key, item in kwargs.items()])

    cmd = (
        f"calibrate "
        f"-datacolumn {ms.column} "
        f"-m {str(calibrate_model)} "
        f"{calibrate_kwargs} "
        f"{str(ms.path)} "
        f"{str(solution_path)} "
    )

    logger.debug(f"Constructed calibrate command is {cmd=}")

    return CalibrateCommand(cmd=cmd, solution_path=solution_path, ms=ms)


def create_apply_solutions_cmd(
    ms: MS, solutions_file: Path, output_column: Optional[str] = None
) -> ApplySolutionsCommand:
    """Construct the command to apply calibration solutions to a MS
    using an AO calibrate style solutions file.

    The `applysolutions` program does not appear to have the ability to set
    a desured output column name. If the `output_column` specified matches
    the nominated column in `ms`, then `applysolutions` will simply overwrite
    the column with updated data. Otherwise, a `CORRECTED_DATA` column is produced.

    NOTE: Care to be taken when the nominated column is `CORRECTED_DATA`.

    Args:
        ms (MS): Measurement set to have solutions applied to
        solutions_file (Path): Path to the solutions file to apply
        output_column (Optional[str], optional): The desired output column name. See notes above. Defaults to None.

    Returns:
        ApplySolutionsCommand: _description_
    """

    assert ms.path.exists(), f"The measurement set {ms} was not found. "
    assert ms.column is not None, f"{ms} does not have a nominated data_column. "
    assert (
        solutions_file.exists()
    ), f"The solutions file {solutions_file} does not exists. "

    input_column = ms.column
    copy_mode = "-nocopy" if input_column == output_column else "-copy"

    logger.info(f"Setting {copy_mode=}.")

    if copy_mode == "-copy":
        output_column = "CORRECT_DATA"

    cmd = (
        f"applysolutions "
        f"-datacolumn {input_column} "
        f"{copy_mode} "
        f"{str(ms.path)} "
        f"{str(solutions_file)} "
    )

    logger.info(f"Constructed {cmd=}")

    return ApplySolutionsCommand(
        cmd=cmd, solution_path=solutions_file, ms=ms.with_options(column=output_column)
    )


def run_calibrate(calibrate_cmd: CalibrateCommand, container: Path) -> None:
    """Execute a calibrate command within a singularity container

    Args:
        calibrate_cmd (CalibrateCommand): The constructed calibrate command
        container (Path): Location of the container
    """

    assert container.exists(), f"The calibrate container {container} does not exist. "
    assert (
        calibrate_cmd.ms is not None
    ), f"When calibrating the 'ms' field attribute must be defined. "

    run_singularity_command(
        image=container,
        command=calibrate_cmd.cmd,
        bind_dirs=[calibrate_cmd.solution_path.parent, calibrate_cmd.ms.path.parent],
    )


def run_apply_solutions(
    apply_solutions_cmd: ApplySolutionsCommand, container: Path
) -> None:
    """Will execute the applysolutions command inside the specified singularity
    container.

    Args:
        apply_solutions_cmd (ApplySolutionsCommand): The constructed applysolutions command
        container (Path): Location of the existing solutions file
    """

    assert (
        container.exists()
    ), f"The applysolutions container {container} does not exist. "
    assert (
        apply_solutions_cmd.ms.path.exists()
    ), f"The measurement set {apply_solutions_cmd.ms} was not found. "

    run_singularity_command(
        image=container,
        command=apply_solutions_cmd.cmd,
        bind_dirs=[
            apply_solutions_cmd.solution_path.parent,
            apply_solutions_cmd.ms.path.parent,
        ],
    )


def calibrate_apply_ms(
    ms_path: Path, model_path: Path, container: Path, data_column: str = "DATA"
) -> MS:
    """Will create and run a calibration command using AO calibrator, and then appy these solutions.

    Args:
        ms_path (Path): The measurement set that will be calibrated
        model_path (Path): The model file containing sources to calibrate against
        container (Path): Container that has the AO calibtate and applysolutions file.
        data_column (str, optional): The name of the column containing the data to calibrate. Defaults to "DATA".

    Returns:
        MS: Measurement set with the column attribute set to the column with corrected data
    """
    ms = MS(path=ms_path, column=data_column)

    logger.info(f"Will be attempting to calibrate {ms}")

    calibrate_cmd = create_calibrate_cmd(ms=ms, calibrate_model=model_path)

    run_calibrate(calibrate_cmd=calibrate_cmd, container=container.absolute())

    apply_solutions_cmd = create_apply_solutions_cmd(
        ms=ms, solutions_file=calibrate_cmd.solution_path
    )

    run_apply_solutions(
        apply_solutions_cmd=apply_solutions_cmd, container=container.absolute()
    )

    return apply_solutions_cmd.ms


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Run calibrate and apply the solutions given a measurement set and sky-model."
    )

    parser.add_argument(
        "ms",
        type=Path,
        help="The measurement set to calibrate and apply solutions to. ",
    )
    parser.add_argument(
        "aoskymodel",
        type=Path,
        help="The AO-style sky-model file to use when calibrating. ",
    )
    parser.add_argument(
        "--calibrate-container",
        type=Path,
        default="./calibrate.sif",
        help="The container containing calibrate and applysolutions. ",
    )
    parser.add_argument(
        "--data-column", type=str, default="DATA", help="The column to calibrate"
    )

    return parser


def cli() -> None:
    import logging

    parser = get_parser()

    args = parser.parse_args()

    logger.setLevel(logging.DEBUG)

    calibrate_apply_ms(
        ms_path=args.ms,
        model_path=args.aoskymodel,
        container=args.calibrate_container,
        data_column=args.data_column,
    )


if __name__ == "__main__":
    cli()
