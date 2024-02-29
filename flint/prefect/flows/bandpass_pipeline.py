"""Pipeline to calibrate a ASKAP style bandpass observation.

At the moment this is expected to be performed against observations taken
towards PKS1934-638. At the time of writing a typical ASKAP bandpass
observation will cycle each beam so that it is centred on this source.
This means that practically there are 36 separate fields at slightly
different field centres. The bandpass calibration process will first have
to split the correct field out before actually calibration.
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Collection, List, Optional

from prefect import flow, task, unmapped

from flint.bandpass import extract_correct_bandpass_pointing
from flint.calibrate.aocalibrate import (
    ApplySolutions,
    CalibrateCommand,
    create_apply_solutions_cmd,
    create_calibrate_cmd,
    flag_aosolutions,
    select_aosolution_for_ms,
)
from flint.flagging import flag_ms_aoflagger
from flint.logging import logger
from flint.ms import MS, preprocess_askap_ms, split_by_field
from flint.naming import get_sbid_from_path
from flint.options import BandpassOptions
from flint.prefect.clusters import get_dask_runner
from flint.prefect.common.utils import upload_image_as_artifact
from flint.sky_model import get_1934_model

# These are generic functions that are wrapped. Their inputs are fairly standard
# and do not require any type of unpacking or testing before they are use.
task_extract_correct_bandpass_pointing = task(extract_correct_bandpass_pointing)
task_preprocess_askap_ms = task(preprocess_askap_ms)
task_flag_ms_aoflagger = task(flag_ms_aoflagger)
task_create_calibrate_cmd = task(create_calibrate_cmd)
task_split_by_field = task(split_by_field)
task_select_solution_for_ms = task(select_aosolution_for_ms)
task_create_apply_solutions_cmd = task(create_apply_solutions_cmd)


# The tasks below are ones that require some of the inputs to be cast or transformed
# into something that is known to the actual worker functions.


@task
def task_bandpass_create_apply_solutions_cmd(
    ms: MS,
    calibrate_cmd: CalibrateCommand,
    container: Path,
    output_column: Optional[str] = None,
) -> ApplySolutions:
    """Apply an ao-calibrate style solutions file to an input measurement set.

    Internally the solutions path to apply to the nominaled measurement set is extracted
    from the incoming ``calibrate_cmd``.

    Args:
        ms (MS): The measurement set that will have solutions applied
        calibrate_cmd (CalibrateCommand): The calibrate command and meta-data describing the solutions to apply
        container (Path): Path to singularity container that will apply the solutions
        output_column (Optional[Path], optional): the output column anme to create. Defaults to None.

    Returns:
        ApplySolutions: The apply solutions command and meta-data
    """
    return create_apply_solutions_cmd(
        ms=ms,
        solutions_file=calibrate_cmd.solution_path,
        output_column=output_column,
        container=container,
    )


@task
def task_flag_solutions(
    calibrate_cmd: CalibrateCommand,
    smooth_window_size: int = 16,
    smooth_polynomial_order: int = 4,
    **kwargs,
) -> CalibrateCommand:
    """Flag calibration solutions

    Args:
        calibrate_cmd (CalibrateCommand): Calibrate command that contains path to the solution file that will be flagged
        smooth_window_size (int, optional): The size of the window function of the savgol filter. Passed directly to savgol. Defaults to 16.
        smooth_polynomial_order (int, optional): The order of the polynomial of the savgol filter. Passed directly to savgol. Defaults to 4.

    Returns:
        CalibrateCommand: Calibrate command with update meta-data describing the new solutions file
    """
    solution_path = calibrate_cmd.solution_path
    ms_path = calibrate_cmd.ms.path

    plot_dir = ms_path.parent / Path("preflagger")
    if not plot_dir.exists():
        try:
            logger.info(f"Attempting to create {plot_dir}")
            plot_dir.mkdir(parents=True)
        except FileExistsError:
            logger.warning(
                "Creating the directory failed. Likely already exists. Race conditions, me-hearty."
            )

    flagged_solutions = flag_aosolutions(
        solutions_path=solution_path,
        ref_ant=-1,
        flag_cut=3,
        plot_dir=plot_dir,
        smooth_solutions=True,
        smooth_window_size=smooth_window_size,
        smooth_polynomial_order=smooth_polynomial_order,
        **kwargs,
    )

    for image_path in flagged_solutions.plots:
        upload_image_as_artifact(image_path=image_path, description=image_path.name)

    return calibrate_cmd.with_options(
        solution_path=flagged_solutions.path, preflagged=True
    )


def run_bandpass_stage(
    bandpass_mss: Collection[MS],
    output_split_bandpass_path: Path,
    bandpass_options: BandpassOptions,
    model_path: Path,
    source_name_prefix: str = "B1934-638",
    skip_rotation: bool = False,
) -> List[CalibrateCommand]:
    """Excutes the bandpass calibration (using ``calibrate``) against a set of
    input measurement sets.

    Args:
        bandpass_mss (Collection[MS]): Set of bandpass measurement sets to calibrate
        output_split_bandpass_path (Path): The location where the extract field centred on the calibration field (typically PKSB19340638)
        bandpass_options (BandpassOptions): Configurables that will specify the bandpass calibbration process
        model_path (Path): Path to the model used to calibrate against
        source_name_prefix (str, optional): Name of the field being calibrated. Defaults to "B1934-638".
        skip_rotation (bool, optional): If ``True`` the rotation of the ASKAP visibility from the antenna frame to the sky-frame will be skipped. Defaults to False.

    Returns:
        List[CalibrateCommand]: Set of calibration commands used
    """
    assert (
        bandpass_options.flag_calibrate_rounds >= 0
    ), f"Currently {bandpass_options.flag_calibrate_rounds=}, needs to be 0 or higher"

    if not output_split_bandpass_path.exists():
        logger.info(f"Creating {str(output_split_bandpass_path)}")
        output_split_bandpass_path.mkdir(parents=True)

    calibrate_cmds: List[CalibrateCommand] = []

    extract_bandpass_mss = task_extract_correct_bandpass_pointing.map(
        ms=bandpass_mss,
        source_name_prefix=source_name_prefix,
        ms_out_dir=output_split_bandpass_path,
    )
    preprocess_bandpass_mss = task_preprocess_askap_ms.map(
        ms=extract_bandpass_mss, skip_rotation=skip_rotation
    )
    flag_bandpass_mss = task_flag_ms_aoflagger.map(
        ms=preprocess_bandpass_mss,
        container=bandpass_options.flagger_container,
        rounds=1,
    )
    calibrate_cmds = task_create_calibrate_cmd.map(
        ms=flag_bandpass_mss,
        calibrate_model=model_path,
        container=bandpass_options.calibrate_container,
        update_calibrate_options=unmapped(dict(minuv=bandpass_options.minuv)),
    )

    for i in range(bandpass_options.flag_calibrate_rounds):
        # Apply and then recalibrate
        apply_cmds = task_bandpass_create_apply_solutions_cmd.map(
            ms=calibrate_cmds,
            calibrate_cmd=calibrate_cmds,
            output_column="CORRECTED_DATA",
            container=bandpass_options.calibrate_container,
        )
        flag_bandpass_mss = task_flag_ms_aoflagger.map(
            ms=apply_cmds, container=bandpass_options.flagger_container, rounds=1
        )
        calibrate_cmds = task_create_calibrate_cmd.map(
            ms=flag_bandpass_mss,
            calibrate_model=model_path,
            container=bandpass_options.calibrate_container,
            calibrate_data_column="DATA",
            update_calibrate_options=unmapped(dict(minuv=bandpass_options.minuv)),
        )
    flag_calibrate_cmds = task_flag_solutions.map(
        calibrate_cmd=calibrate_cmds,
        smooth_window_size=bandpass_options.smooth_window_size,
        smooth_polynomial_order=bandpass_options.smooth_polynomial_order,
        mean_ant_tolerance=bandpass_options.preflagger_ant_mean_tolerance,
        mesh_ant_flags=bandpass_options.preflagger_mesh_ant_flags,
    )

    return flag_calibrate_cmds


@flow
def calibrate_bandpass_flow(
    bandpass_path: Path,
    split_path: Path,
    bandpass_options: BandpassOptions,
) -> Path:
    """Create and run the prefect flow to calibrate a set of bandpass measurement sets.

    The measurement sets that will be calibreated are expected to:
    - be following the raw name format convention
    - reside in a directory whose name is the SBID of the observation

    The current bandpass procedure currently relies on the Andre Offringa's ``calibrate`` tool,
    with slight modification from Emil Lenc. The well known source PKS1934-638 is the only source
    supported for bandpass calibration, and its model is packaged inside ``flint``. This model
    is in the AO-style format.

    Each measurement set will correspond to a solutions file once the ``calibrate`` tool has been
    executed successfully. These should be kept together -- there is not enough meta-data in the
    single solutions file to denote the frequency / channels / beam number described in the measurement set.

    Args:
        bandpass_path (Path): Location to the folder containing the raw ASKAP bandpass measurement sets that will be calibrated
        split_path (Path): Location that will contain a folder, named after the SBID of the observation, that will contain the output bandpass measurement sets, solutions and plots
        bandpass_options (BandpassOptions): Options that specify configurables of the bandpass processing.

    Returns:
        Path: Directory that contains the extracted measurement sets and the ao-style gain solutions files.
    """
    assert (
        bandpass_path.exists() and bandpass_path.is_dir()
    ), f"{str(bandpass_path)} does not exist or is not a folder. "
    bandpass_mss = list([MS.cast(ms_path) for ms_path in bandpass_path.glob("*.ms")])

    assert (
        len(bandpass_mss) == bandpass_options.expected_ms
    ), f"Expected to find {bandpass_options.expected_ms} in {str(bandpass_path)}, found {len(bandpass_mss)}."

    logger.info(
        f"Found the following bandpass measurement set: {[bp.path for bp in bandpass_mss]}."
    )

    bandpass_folder_name = bandpass_path.name
    output_split_bandpass_path = (
        Path(split_path / bandpass_folder_name).absolute().resolve()
    )
    logger.info(
        f"Will write extracted bandpass MSs to: {str(output_split_bandpass_path)}."
    )

    # This is the model that we will calibrate the bandpass against.
    # At the time fo writing 1934-638 is the only model that is supported,
    # not only by this pirate ship, but also the ASKAP telescope itself.
    model_path: Path = get_1934_model(mode="calibrate")
    source_name_prefix: str = "B1934-638"

    run_bandpass_stage(
        bandpass_mss=bandpass_mss,
        output_split_bandpass_path=output_split_bandpass_path,
        bandpass_options=bandpass_options,
        model_path=model_path,
        source_name_prefix=source_name_prefix,
        skip_rotation=True,
    )

    return output_split_bandpass_path


def setup_run_bandpass_flow(
    bandpass_path: Path,
    split_path: Path,
    cluster_config: Path,
    bandpass_options: BandpassOptions,
) -> Path:
    """Create and run the prefect flow to calibrate a set of bandpass measurement sets.

    The measurement sets that will be calibreated are expected to:
    - be following the raw name format convention
    - reside in a directory whose name is the SBID of the observation

    The current bandpass procedure currently relies on the Andre Offringa's ``calibrate`` tool,
    with slight modification from Emil Lenc. The well known source PKS1934-638 is the only source
    supported for bandpass calibration, and its model is packaged inside ``flint``. This model
    is in the AO-style format.

    Each measurement set will correspond to a solutions file once the ``calibrate`` tool has been
    executed successfully. These should be kept together -- there is not enough meta-data in the
    single solutions file to denote the frequency / channels / beam number described in the measurement set.

    Args:
        bandpass_path (Path): Location to the folder containing the raw ASKAP bandpass measurement sets that will be calibrated
        split_path (Path): Location that will contain a folder, named after the SBID of the observation, that will contain the output bandpass measurement sets, solutions and plots
        cluster_config (Path): Path to a yaml file that is used to configure a prefect dask task runner.
        bandpass_options (BandpassOptions): Options that specify configurables of the bandpass processing.

    Returns:
        Path: Directory that contains the extracted measurement sets and the ao-style gain solutions files.
    """

    dask_task_runner = get_dask_runner(cluster=cluster_config)

    bandpass_sbid = get_sbid_from_path(path=bandpass_path)

    calibrate_bandpass_flow.with_options(
        name=f"Flint Bandpass Pipeline -- {bandpass_sbid}", task_runner=dask_task_runner
    )(
        bandpass_path=bandpass_path,
        split_path=split_path,
        bandpass_options=bandpass_options,
    )

    return bandpass_path


def get_parser() -> ArgumentParser:
    """Create an argument paraser for the bandpass prefect workflow

    Returns:
        ArgumentParser: CLI argument parser
    """
    parser = ArgumentParser(
        description="Perform bandpass calibration against an ASKAP SBID. "
    )

    parser.add_argument(
        "bandpass_path",
        type=Path,
        help="Path to the directory containing the uncalibrated bandpass measurement sets. ",
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        default=Path("."),
        help="Location to write the field-split MSs. Will attempt to create a directory using the SBID of the bandpass observation. ",
    )

    parser.add_argument(
        "--expected-ms",
        type=int,
        default=36,
        help="The expected number of measurement sets to find. ",
    )
    parser.add_argument(
        "--calibrate-container",
        type=Path,
        default="aocalibrate.sif",
        help="Path to container that holds AO calibrate and applysolutions. ",
    )
    parser.add_argument(
        "--flagger-container",
        type=Path,
        default="flagger.sif",
        help="Path to container with aoflagger software. ",
    )
    parser.add_argument(
        "--cluster-config",
        type=str,
        default="petrichor",
        help="Path to a cluster configuration file, or a known cluster name. ",
    )
    parser.add_argument(
        "--smooth-window-size",
        default=16,
        type=int,
        help="Size of the smoothing Savgol window when smoothing bandpass solutions",
    )
    parser.add_argument(
        "--smooth-polynomial-order",
        default=4,
        type=int,
        help="Order of the polynomial when smoothing the bandpass solutions with the Savgol filter",
    )
    parser.add_argument(
        "--flag-calibrate-rounds",
        type=int,
        default=3,
        help="The number of times a bandpass solution will be derived, applied and flagged. ",
    )
    parser.add_argument(
        "--minuv",
        type=float,
        default=None,
        help="The minimum baseline length, in meters, for data to be included in bandpass calibration stage",
    )
    parser.add_argument(
        "--preflagger-ant-mean-tolerance",
        type=float,
        default=0.2,
        help="Tolerance of the mean x/y antenna gain ratio test before antenna is flagged",
    )
    parser.add_argument(
        "--preflagger-mesh-ant-flags",
        default=False,
        action="store_true",
        help="Share channel flags from bandpass solutions between all antennas",
    )

    return parser


def cli() -> None:
    import logging

    logger.setLevel(logging.INFO)

    parser = get_parser()

    args = parser.parse_args()

    bandpass_options = BandpassOptions(
        flagger_container=args.flagger_container,
        calibrate_container=args.calibrate_container,
        expected_ms=args.expected_ms,
        smooth_window_size=args.smooth_window_size,
        smooth_polynomial_order=args.smooth_polynomial_order,
        flag_calibrate_rounds=args.flag_calibrate_rounds,
        minuv=args.minuv,
        preflagger_ant_mean_tolerance=args.preflagger_ant_mean_tolerance,
        preflagger_mesh_ant_flags=args.preflagger_mesh_ant_flags,
    )

    setup_run_bandpass_flow(
        bandpass_path=args.bandpass_path,
        split_path=args.split_path,
        cluster_config=args.cluster_config,
        bandpass_options=bandpass_options,
    )


if __name__ == "__main__":
    cli()
