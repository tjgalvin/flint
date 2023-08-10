"""Code to use AO calibrate s
"""
from __future__ import annotations  # used to keep mypy/pylance happy in AOSolutions
import struct
from pathlib import Path
from typing import NamedTuple, Optional, Union, Iterable, List, Collection, List
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
from spython.main import Client as sclient

from flint.logging import logger
from flint.ms import MS, get_beam_from_ms, consistent_ms
from flint.sclient import run_singularity_command
from flint.plot_utils import fill_between_flags
from flint.bptools.preflagger import (
    flag_outlier_phase,
    flags_over_threshold,
    flag_mean_residual_amplitude,
    flag_mean_xxyy_amplitude_ratio,
)
from flint.exceptions import PhaseOutlierFitError

CALIBRATE_SUFFIX = ".calibrate.bin"
PREFLAGGED_SUFFIX = ".preflagged"


class CalibrateCommand(NamedTuple):
    """The AO Calibrate command and output path of the corresponding solutions file"""

    cmd: str
    """The calibrate command that will be executed
    """
    solution_path: Path
    """The output path of the solutions file
    """
    ms: MS
    """The measurement set to have solutions derived for"""
    model: Path
    """Path to the model that would be used to calibrate against"""
    preflagged: bool = False
    """Indicates whether the solution file has gone through preflagging routines. """

    def with_options(self, **kwargs) -> CalibrateCommand:
        _dict = self._asdict()
        _dict.update(**kwargs)

        return CalibrateCommand(**_dict)


class ApplySolutions(NamedTuple):
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

    # TODO: Need tocorporate the start and end times into this header

    @classmethod
    def load(cls, path: Path) -> AOSolutions:
        """Load in an AO-stule solution file. See `load_solutions_file`, which is
        internally used.
        """
        return load_aosolutions_file(solutions_path=path)

    def save(self, output_path: Path) -> Path:
        """Save the instance of AOSolution to a standard aosolution binary file

        Args:
            output_path (Path): Location to write the file to

        Returns:
            Path: Location the file was written to
        """
        return save_aosolutions_file(aosolutions=self, output_path=output_path)

    def plot_solutions(self, ref_ant: int = 0) -> Iterable[Path]:
        """Plot the solutions of all antenna for the first time-inteval
        in the aosolutions file. The XX and the YY will be plotted.

        Args:
            ref_ant (int, optional): Reference antenna to divide the solutions by. Defaults to 0.

        Returns:
            Iterable[Path]: Path to the phase and amplited plots created.
        """
        # TODO: Change call signature to pass straight through
        return plot_solutions(solutions=self, ref_ant=ref_ant)


def plot_solutions(
    solutions: Union[Path, AOSolutions], ref_ant: int = 0
) -> Collection[Path]:
    """Plot solutions for AO-style solutions

    Args:
        solutions (Path): Path to the solutions file
        ref_ant (int, optional): Reference antenna to use. Defaults to 0.

    Return:
        Collection[Path] -- The paths of the two plots createda
    """
    ao_sols = (
        AOSolutions.load(path=solutions) if isinstance(solutions, Path) else solutions
    )
    solutions_path = ao_sols.path
    logger.info(f"Plotting {solutions_path}")

    if ao_sols.nsol > 1:
        logger.warn(f"Found {ao_sols.nsol} intervals, plotting the first. ")
    plot_sol = 0  # The first time interval

    data = ao_sols.bandpass[plot_sol] / ao_sols.bandpass[plot_sol, ref_ant, :, :]
    amplitudes = np.abs(data)
    phases = np.angle(data, deg=True)
    channels = np.arange(ao_sols.nchan)

    ncolumns = 6
    nrows = ao_sols.nant // ncolumns
    if ncolumns * nrows < ao_sols.nant:
        nrows += 1
    logger.debug(f"Plotting {plot_sol=} with {ncolumns=} {nrows=}")

    fig_amp, axes_amp = plt.subplots(nrows, ncolumns, figsize=(14, 7))
    fig_phase, axes_phase = plt.subplots(nrows, ncolumns, figsize=(14, 7))

    for y in range(nrows):
        for x in range(ncolumns):
            ant = y * nrows + x

            amps_xx = amplitudes[ant, :, 0]
            amps_yy = amplitudes[ant, :, 3]
            phase_xx = phases[ant, :, 0]
            phase_yy = phases[ant, :, 3]

            if any([np.sum(~np.isfinite(amps)) == 0 for amps in (amps_xx, amps_yy)]):
                logger.warn(f"No valid data for {ant=}")
                continue

            max_amp_xx = (
                np.nanmax(amps_xx[np.isfinite(amps_xx)])
                if any(np.isfinite(amps_xx))
                else -1
            )
            max_amp_yy = (
                np.nanmax(amps_yy[np.isfinite(amps_yy)])
                if any(np.isfinite(amps_yy))
                else -1
            )

            max_amp = np.nanmax([max_amp_xx, max_amp_yy])
            ax_a, ax_p = axes_amp[y, x], axes_phase[y, x]
            ax_a.plot(channels, amps_xx, marker=None, color="blue")
            ax_a.plot(channels, amps_yy, marker=None, color="red")
            ax_a.set(ylim=(0, 1.2 * max_amp))
            ax_a.set_title(f"ant{ant:02d}", fontsize=8)
            fill_between_flags(ax_a, ~np.isfinite(amps_yy) | ~np.isfinite(amps_xx))

            ax_p.plot(channels, phase_xx, marker=None, color="blue")
            ax_p.plot(channels, phase_yy, marker=None, color="red")
            ax_p.set(ylim=(-200, 200))
            ax_p.set_title(f"ak{ant:02d}", fontsize=8)
            fill_between_flags(ax_p, ~np.isfinite(phase_yy) | ~np.isfinite(phase_xx))

    fig_amp.suptitle(f"{ao_sols.path.name} - Amplitudes")
    fig_phase.suptitle(f"{ao_sols.path.name} - Phases")

    fig_amp.tight_layout()
    fig_phase.tight_layout()

    out_amp = f"{str(solutions_path.with_suffix('.amplitude.pdf'))}"
    logger.info(f"Saving {out_amp}.")
    fig_amp.savefig(out_amp)

    out_phase = f"{str(solutions_path.with_suffix('.phase.pdf'))}"
    logger.info(f"Saving {out_phase}.")
    fig_phase.savefig(out_phase)

    return [Path(out_amp), Path(out_phase)]


def save_aosolutions_file(aosolutions: AOSolutions, output_path: Path) -> Path:
    """Save a AOSolutions file to the ao-standard binary format.

    Args:
        aosolutions (ApplySolutions): Instance of the solutions to save
        output_path (Path): Output path to write the files to

    Returns:
        Path: Path the file was written to
    """

    header_format = "8s6I2d"
    header_intro = b"MWAOCAL\0"

    output_dir = output_path.parent
    if not output_dir.exists():
        logger.info(f"Creating {output_dir}.")
        output_dir.mkdir(parents=True)

    logger.info(f"Writing aosolutions to {str(output_path)}.")
    with open(str(output_path), "wb") as out_file:
        out_file.write(
            struct.pack(
                header_format,
                header_intro,
                0,  # File type, only 0 mode available
                0,  # Structure type, 0 model available only
                aosolutions.nsol,
                aosolutions.nant,
                aosolutions.nchan,
                aosolutions.npol,
                0.0,  # time start, I don't believe these are used in most use cases
                0.0,  # time end, I don't believe these are used in most use cases
            )
        )
        aosolutions.bandpass.tofile(out_file)

    return output_path


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

        bandpass = np.fromfile(in_file, dtype="<c16", count=np.prod(sol_shape)).reshape(
            sol_shape
        )
        logger.info(f"Loaded solutions of shape {bandpass.shape}")

        return AOSolutions(
            path=solutions_path,
            nsol=nsol,
            nant=nant,
            nchan=nchan,
            npol=npol,
            bandpass=bandpass,
        )


def find_existing_solutions(
    bandpass_directory: Path,
    expected_suffix: Optional[str] = None,
    use_preflagged: bool = True,
    model_path: Path = Path("."),
) -> List[CalibrateCommand]:
    """Given a directory that contains a collection of bandpass measurement
    sets, attempt to identify a corresponding set of calibrate binary solution
    file.

    This search only supports the use of the default or known preflagger suffix.
    Limited support is provided to specify the expected calibrate suffix.

    These bandpass measurement sets should be processed already - which means just
    the B1936-638 field has been split out of the larger raw MS, flagged and
    calibrated. These steps are expected in order to get to the calibrate
    stage.

    Args:
        bandpass_directory (Path): Directory to search for split bandpass measurement sets
        expected_suffix (Optional[str], optional): The suffix to use to find the calibration binary files. If None, a known format is used. Defaults to None.
        use_preflagged (bool, optional): Add the pre-flag suffix when searching for solution files. This uses the expected suffix for preflagged solutions. Defaults to True.
        model_path (Path, optional): Path to the model file to use. This is passed through to create the CalibrateCommand, and is not strictly required. Defaults to Path('.').

    Returns:
        List[CalibrateCommand]: Collection of the calibrate command strcutures that are intended to be used to map the bandpass measurement sets to solution files.
    """

    logger.info(
        f"Searching {bandpass_directory} for existing measurement sets and solutions. "
    )
    expected_suffix = expected_suffix if expected_suffix else CALIBRATE_SUFFIX

    if use_preflagged:
        logger.info(f"Will search for preflagged solutions. ")
        expected_suffix += ".preflagged"

    bandpass_mss = bandpass_directory.glob("*ms")

    # This is a placeholder command. It will not be executed,
    # but is used to link the measurement set with the appropriate
    # solutions file. At least with ao-calibrate, there is not
    # enough information to specify frequency of channels, time
    # etc. Matching the solutions to the original MS is the
    # safest thing to do, particularly with the step of matching
    # the solutions to the beam images. The DATA column is
    # only relevant for the calibration command, and not used
    # in any other capacity, ya rotten scallywag
    calibrate_cmds = [
        create_calibrate_cmd(ms=MS(path=ms, column="DATA"), calibrate_model=model_path)
        for ms in bandpass_mss
    ]

    if use_preflagged:
        logger.info(
            f"Updating solution files with preflagger suffix {PREFLAGGED_SUFFIX}."
        )
        calibrate_cmds = [
            calibrate_cmd.with_options(
                preflagged=True,
                solution_path=calibrate_cmd.solution_path.with_suffix(
                    PREFLAGGED_SUFFIX
                ),
            )
            for calibrate_cmd in calibrate_cmds
        ]

    logger.info(f"Constructed Calibrate command list of length {len(calibrate_cmds)}")

    # If not all the treasure could be found. At the moment this function will only
    # work if the bandpass solutions were made using the default values.
    assert all(
        [calibrate_cmd.solution_path.exists() for calibrate_cmd in calibrate_cmds]
    ), f"Missing solution file constructed from scanning {bandpass_directory}. Check the directory. "

    return calibrate_cmds


def select_aosolution_for_ms(
    calibrate_cmds: List[CalibrateCommand], ms: Union[MS, Path]
) -> Path:
    """Attempt to select an AO-style solution file for a measurement
    set. This can be expanded to include a number of criteria, but
    at present it only searches for a matching beam number between
    the input set of CalibrationCommands and the input MS.

    Args:
        calibrate_cmds (List[CalibrateCommand]): Set of calibration commands, which includes the solution file path and the corresponding MS, as attributes.
        ms (Union[MS, Path]): The measurement sett that needs a solutions file.

    Raises:
        ValueError: Raised when not matching AO-solution file found.

    Returns:
        Path: Path to solution file to apply.
    """
    ms = MS.cast(ms)
    ms_beam = ms.beam if ms.beam is not None else get_beam_from_ms(ms=ms)

    logger.info(f"Will select a solution for {str(ms.path)}, {ms_beam=}.")
    logger.info(f"{len(calibrate_cmds)} potential solutions to consider. ")

    for calibrate_cmd in calibrate_cmds:
        logger.info(f"Considering {str(calibrate_cmd.solution_path)}.")
        if consistent_ms(ms1=ms, ms2=calibrate_cmd.ms):
            sol_file = calibrate_cmd.solution_path
            break
    else:
        raise ValueError(
            f"No solution file found for {str(ms.path)} from {[c.ms.path for c in calibrate_cmds]} found. "
        )

    logger.info(f"Have selected {str(sol_file)} for {str(ms.path)}. ")
    return sol_file


def create_calibrate_cmd(
    ms: Union[Path, MS],
    calibrate_model: Path,
    solution_path: Optional[Path] = None,
    container: Optional[Path] = None,
    **kwargs,
) -> CalibrateCommand:
    """Generate a typical ao calibrate command. Any extra keyword arguments
    are passed through as additional options to the `calibrate` program.

    Args:
        ms (Union[Path,MS]): The measurement set to calibrate. There needs to be a nominated data_column.
        calibrate_model (Path): Path to a generated calibrate sky-model
        solution_path (Path, optional): The output path of the calibrate solutions file. If None, a default suffix of "calibrate.bin" is used. Defaults to None.
        container (Optional[Path], optional): If a path to a container is supplied the calibrate command is executed immediatedly. Defaults to None.

    Raises:
        FileNotFoundError: Raised when calibrate_model can not be found.

    Returns:
        CalibrateCommand: The calibrate command to execute and output solution file
    """
    ms = MS.cast(ms)
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
        f"-minuv 200 "
        f"-m {str(calibrate_model)} "
        f"{calibrate_kwargs} "
        f"{str(ms.path)} "
        f"{str(solution_path)} "
    )

    logger.debug(f"Constructed calibrate command is {cmd=}")

    calibrate_cmd = CalibrateCommand(
        cmd=cmd, solution_path=solution_path, ms=ms, model=calibrate_model
    )

    if container is not None:
        run_calibrate(calibrate_cmd=calibrate_cmd, container=container)

    return calibrate_cmd


def create_apply_solutions_cmd(
    ms: MS,
    solutions_file: Path,
    output_column: Optional[str] = None,
    container: Optional[Path] = None,
) -> ApplySolutions:
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
        container (Optional[Path], optional): If a path to a container is supplied the calibrate command is executed immediatedly. Defaults to None.

    Returns:
        ApplySolutions: Description of applysolutions command, solutions file path and updated MS
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
        output_column = "CORRECTED_DATA"

    cmd = (
        f"applysolutions "
        f"-datacolumn {input_column} "
        f"{copy_mode} "
        f"{str(ms.path)} "
        f"{str(solutions_file)} "
    )

    logger.info(f"Constructed {cmd=}")

    apply_solutions = ApplySolutions(
        cmd=cmd, solution_path=solutions_file, ms=ms.with_options(column=output_column)
    )

    if container is not None:
        run_apply_solutions(apply_solutions_cmd=apply_solutions, container=container)

    return apply_solutions


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
        bind_dirs=[
            calibrate_cmd.solution_path.parent,
            calibrate_cmd.ms.path.parent,
            calibrate_cmd.model.parent,
        ],
    )


def run_apply_solutions(apply_solutions_cmd: ApplySolutions, container: Path) -> None:
    """Will execute the applysolutions command inside the specified singularity
    container.

    Args:
        apply_solutions_cmd (ApplySolutions): The constructed applysolutions command
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
            apply_solutions_cmd.solution_path.parent.absolute(),
            apply_solutions_cmd.ms.path.parent.absolute(),
        ],
    )


def calibrate_apply_ms(
    ms_path: Path, model_path: Path, container: Path, data_column: str = "DATA"
) -> ApplySolutions:
    """Will create and run a calibration command using AO calibrator, and then appy these solutions.

    Args:
        ms_path (Path): The measurement set that will be calibrated
        model_path (Path): The model file containing sources to calibrate against
        container (Path): Container that has the AO calibtate and applysolutions file.
        data_column (str, optional): The name of the column containing the data to calibrate. Defaults to "DATA".

    Returns:
        Applysolutions: The command, solution binary path and new measurement set structure
    """
    ms = MS(path=ms_path, column=data_column)

    logger.info(f"Will be attempting to calibrate {ms}")

    calibrate_cmd = create_calibrate_cmd(ms=ms, calibrate_model=model_path)

    run_calibrate(calibrate_cmd=calibrate_cmd, container=container.absolute())

    flagged_solutions_path = flag_aosolutions(
        solutions_path=calibrate_cmd.solution_path,
        ref_ant=0,
        plot_dir=Path(ms_path.parent) / Path("preflagger"),
    )

    apply_solutions_cmd = create_apply_solutions_cmd(
        ms=ms, solutions_file=flagged_solutions_path
    )

    run_apply_solutions(
        apply_solutions_cmd=apply_solutions_cmd, container=container.absolute()
    )

    return apply_solutions_cmd


def apply_solutions_to_ms(
    ms: Union[Path, MS],
    solutions_path: Path,
    container: Path,
    data_column: str = "DATA",
) -> ApplySolutions:
    ms = ms if isinstance(ms, MS) else MS(path=ms, column=data_column)
    logger.info(f"Will attempt to apply {str(solutions_path)} to {str(ms.path)}.")

    apply_solutions_cmd = create_apply_solutions_cmd(
        ms=ms, solutions_file=solutions_path
    )

    run_apply_solutions(
        apply_solutions_cmd=apply_solutions_cmd, container=container.absolute()
    )

    return apply_solutions_cmd


def flag_aosolutions(
    solutions_path: Path,
    ref_ant: int = 0,
    flag_cut: float = 3,
    plot_dir: Optional[Path] = None,
    out_solutions_path: Optional[Path] = None,
) -> Path:
    """Will open a previously solved ao-calibrate solutions file and flag additional channels and antennae.

    There are currently two main stages. The first will attempt to search for channels where the the phase of the
    gain solution are outliers. The phase over frequency is first unwrapped (delay solved for) before the flagging
    statistics are computed.

    The second stage will flag an entire antenna if more then 80 percent of the flags for a polarisation are flagged.

    Args:
        solutions_path (Path): Location of the solutions file to examine and flag.
        ref_ant (int, optional): Reference antenna to use, which is important when searching for phase-outliers. Defaults to 0.
        flag_cut (float, optional): Significance of a phase-outlier from the mean (or median) before it should be flagged. Defaults to 3.
        plot_dir (Optional[Path], optional): Where diagnostic flagging plots should be written. If None, no plots will be produced. Defaults to None.
        out_solutions_path (Optional[Path], optional): The output path of the flagged solutions file. If None, the solutions_path provided is used. Defaults to None.

    Returns:
        Path: Path to the updated solutions file. This is out_solutions_path if provided, otherwise solutions_path
    """
    solutions = AOSolutions.load(path=solutions_path)
    title = solutions_path.name

    pols = {0: "XX", 1: "XY", 2: "YX", 3: "YY"}

    if plot_dir is not None and not plot_dir.exists():
        logger.info(f"Creating {str(plot_dir)}")
        try:
            plot_dir.mkdir(parents=True)
        except:
            logger.warn(f"Failed to create {str(plot_dir)}.")

    # Note that although the solutions variable (an instance of AOSolutions) is immutable,
    # which includes the reference to the numpy array, the _actual_ numpy array is! So,
    # copying the bandpass below is as we are updating the actual array, which will be
    # written back as a new file later.
    bandpass = solutions.bandpass
    logger.info(f"Loaded bandpass, shape is {bandpass.shape}")

    # TODO: Need a procedure to select an optimal reference antenna. What would
    # happen if the ref_ant was not available when the bandpass was being taken.

    for time in range(solutions.nsol):
        for pol in (0, 3):
            logger.info(f"Processing {pols[pol]} polarisation")
            ref_ant_gains = bandpass[time, ref_ant, :, pol]
            # TODO: A better way of selecting the most appropriate reference antenna is needed.
            if np.sum(np.isfinite(ref_ant_gains)) == 0:
                raise ValueError(f"The ref_ant={ref_ant} is completely bad. ")

            for ant in range(solutions.nant):
                if ant == ref_ant:
                    logger.info(f"Skipping reference antenna = ant{ref_ant:02}")
                    continue

                ant_gains = bandpass[time, ant, :, pol] / ref_ant_gains
                plot_title = f"{title} - ant{ant:02d} - {pols[pol]}"
                ouput_path = (
                    plot_dir / f"{title}.ant{ant:02d}.{pols[pol]}.png"
                    if plot_dir is not None
                    else None
                )

                if np.sum(np.isfinite(ant_gains)) == 0:
                    logger.info(f"Not valid data found for ant{ant:0d} {pols[pol]}")
                    continue

                try:
                    phase_outlier_result = flag_outlier_phase(
                        complex_gains=ant_gains,
                        flag_cut=flag_cut,
                        plot_title=plot_title,
                        plot_path=ouput_path,
                    )
                    bandpass[time, ant, phase_outlier_result.outlier_mask, pol] = np.nan
                except PhaseOutlierFitError:
                    # This is raised if the fit failed to converge, or some other nasty.
                    bandpass[time, ant, :, pol] = np.nan

                # Flag all solutions for this (ant,pol) if more than 80% are flagged
                if flags_over_threshold(
                    flags=~np.isfinite(bandpass[time, ant, :, pol]),
                    thresh=0.8,
                    ant_idx=ant,
                ):
                    logger.info(
                        f"Flagging all solutions across {pols[pol]} for ant{ant:02d}, too many flagged channels."
                    )
                    bandpass[time, ant, :, pol] = np.nan

                complex_gains = bandpass[time, ant, :, pol]
                if any(np.isfinite(complex_gains)) and flag_mean_residual_amplitude(
                    complex_gains=complex_gains
                ):
                    logger.info(
                        f"Flagging all solutions across {pols[pol]} for ant{ant:02d}, mean residual amplitudes high"
                    )
                    bandpass[time, ant, :, pol] = np.nan

                flagged = ~np.isfinite(bandpass[time, ant, :, pol])
                logger.info(
                    f"{ant=:02d}, pol={pols[pol]}, flagged {np.sum(flagged) / ant_gains.shape[0] * 100.:.2f}%"
                )

    # This loop will flag based on stats across different polarisations
    for time in range(solutions.nsol):
        ref_ant_gains = bandpass[time, ref_ant]
        for ant in range(solutions.nant):
            # We need to skip the case of flagging on the reference antenna, I think.
            if ref_ant == ant:
                continue

            ant_gains = bandpass[time, ant] / ref_ant_gains
            if flag_mean_xxyy_amplitude_ratio(
                xx_complex_gains=ant_gains[:, 0], yy_complex_gains=ant_gains[:, 3]
            ):
                logger.info(f"{ant=} failed mean amplitude gain test. Flagging {ant=}.")
                bandpass[time, ant, :, :] = np.nan

    # TODO: This needs to be moved to a common naming module
    out_solutions_path = (
        solutions_path.with_suffix(PREFLAGGED_SUFFIX)
        if out_solutions_path is None
        else out_solutions_path
    )
    solutions.save(output_path=out_solutions_path)

    total_flagged = np.sum(~np.isfinite(bandpass)) / np.prod(bandpass.shape)
    if total_flagged > 0.8:
        msg = (
            f"{total_flagged*100.:.2f}% of {str((solutions_path))} is flagged after running the preflagger. "
            "That is over 90%. "
            f"This surely can not be correct. Likely something has gone very wrong. "
        )
        logger.critical(msg)
        raise ValueError(msg)

    return out_solutions_path


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Run calibrate and apply the solutions given a measurement set and sky-model."
    )

    subparsers = parser.add_subparsers(
        dest="mode", help="AO Calibrate related operations"
    )
    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Calibrate a MS using a text-based sky-model using AO calibrate",
    )

    calibrate_parser.add_argument(
        "ms",
        type=Path,
        help="The measurement set to calibrate and apply solutions to. ",
    )
    calibrate_parser.add_argument(
        "aoskymodel",
        type=Path,
        help="The AO-style sky-model file to use when calibrating. ",
    )
    calibrate_parser.add_argument(
        "--calibrate-container",
        type=Path,
        default="./calibrate.sif",
        help="The container containing calibrate and applysolutions. ",
    )
    calibrate_parser.add_argument(
        "--data-column", type=str, default="DATA", help="The column to calibrate"
    )

    apply_parser = subparsers.add_parser(
        "apply",
        help="Apply an existing AO-style solutions binary to a measurement set. ",
    )

    apply_parser.add_argument(
        "ms", type=Path, help="Path to the measurement set to apply the solutions to. "
    )
    apply_parser.add_argument(
        "aosolutions", type=Path, help="Path to the AO-style binary solutions file. "
    )
    apply_parser.add_argument(
        "--calibrate-container",
        type=Path,
        default="./calibrate.sif",
        help="The container containing calibrate and applysolutions. ",
    )
    apply_parser.add_argument(
        "--data-column", type=str, default="DATA", help="The column to calibrate"
    )

    flag_sols_parser = subparsers.add_parser(
        "flag",
        help="Attempt to flag the bandpass solutions in an ao-style binary solutions file",
    )

    flag_sols_parser.add_argument(
        "aosolutions", type=Path, help="Path to the solution file to inspect and flag"
    )
    flag_sols_parser.add_argument(
        "--flag-cut",
        type=float,
        default=3.0,
        help="The significance level thaat an outlier phase has to be before being flagged",
    )
    flag_sols_parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory to write diagnostic plots to. If unset no plots will be created. ",
    )

    return parser


def cli() -> None:
    import logging

    parser = get_parser()

    args = parser.parse_args()

    logger.setLevel(logging.DEBUG)

    if args.mode == "calibrate":
        calibrate_apply_ms(
            ms_path=args.ms,
            model_path=args.aoskymodel,
            container=args.calibrate_container,
            data_column=args.data_column,
        )
    elif args.mode == "apply":
        apply_solutions_to_ms(
            ms=args.ms,
            solutions_path=args.aosolutions,
            container=args.calibrate_container,
            data_column=args.data_column,
        )
    elif args.mode == "flag":
        flag_aosolutions(
            solutions_path=args.aosolutions,
            flag_cut=args.flag_cut,
            plot_dir=args.plot_dir,
        )


if __name__ == "__main__":
    cli()
