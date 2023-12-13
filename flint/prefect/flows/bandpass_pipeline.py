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
from typing import Collection, List

from prefect import flow, task

from flint.logging import logger
from flint.prefect.clusters import get_dask_runner
from flint.naming import get_sbid_from_path
from flint.bandpass import extract_correct_bandpass_pointing
from flint.calibrate.aocalibrate import (
    CalibrateCommand,
    create_apply_solutions_cmd,
    create_calibrate_cmd,
    flag_aosolutions,
    select_aosolution_for_ms,
)
from flint.ms import MS, preprocess_askap_ms, split_by_field
from flint.flagging import flag_ms_aoflagger
from flint.sky_model import get_1934_model

task_extract_correct_bandpass_pointing = task(extract_correct_bandpass_pointing)
task_preprocess_askap_ms = task(preprocess_askap_ms)
task_flag_ms_aoflagger = task(flag_ms_aoflagger)
task_create_calibrate_cmd = task(create_calibrate_cmd)
task_split_by_field = task(split_by_field)
task_select_solution_for_ms = task(select_aosolution_for_ms)
task_create_apply_solutions_cmd = task(create_apply_solutions_cmd)


@task
def task_bandpass_create_apply_solutions_cmd(
    ms: MS, calibrate_cmd: CalibrateCommand, container: Path
):
    return create_apply_solutions_cmd(
        ms=ms, solutions_file=calibrate_cmd.solution_path, container=container
    )


@task
def task_flag_solutions(calibrate_cmd: CalibrateCommand) -> CalibrateCommand:
    solution_path = calibrate_cmd.solution_path
    ms_path = calibrate_cmd.ms.path

    plot_dir = ms_path.parent / Path("preflagger")
    if not plot_dir.exists():
        try:
            logger.info(f"Attempting to create {plot_dir}")
            plot_dir.mkdir(parents=True)
        except FileExistsError:
            logger.warn(
                "Creating the directory failed. Likely already exists. Race conditions, me-hearty."
            )

    flagged_solutions_path = flag_aosolutions(
        solutions_path=solution_path, ref_ant=0, flag_cut=3, plot_dir=plot_dir
    )

    return calibrate_cmd.with_options(
        solution_path=flagged_solutions_path, preflagged=True
    )


def run_bandpass_stage(
    bandpass_mss: Collection[MS],
    output_split_bandpass_path: Path,
    calibrate_container: Path,
    flagger_container: Path,
    model_path: Path,
    source_name_prefix: str = "B1934-638",
    skip_rotation: bool = False,
) -> List[CalibrateCommand]:
    """Excutes the bandpass calibration (using ``calibrate``) against a set of
    input measurement sets.

    Args:
        bandpass_mss (Collection[MS]): Set of bandpass measurement sets to calibrate
        output_split_bandpass_path (Path): The location where the extract field centred on the calibration field (typically PKSB19340638)
        calibrate_container (Path): Path the a singularity container with the ``calibrate`` software
        flagger_container (Path): Path to the singularity container with the ``aoflagger`` software
        model_path (Path): Path to the model used to calibrate against
        source_name_prefix (str, optional): Name of the field being calibrated. Defaults to "B1934-638".
        skip_rotation (bool, optional): If ``True`` the rotation of the ASKAP visibility from the antenna frame to the sky-frame will be skipped. Defaults to False.

    Returns:
        List[CalibrateCommand]: Set of calibration commands used
    """

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
    flag_bandpass_mss = task_flag_ms_aoflagger.submit(
        ms=preprocess_bandpass_mss, container=flagger_container, rounds=1
    )
    calibrate_cmds = task_create_calibrate_cmd.submit(
        ms=flag_bandpass_mss,
        calibrate_model=model_path,
        container=calibrate_container,
    )

    return calibrate_cmds


@flow
def calibrate_bandpass_flow(
    bandpass_path: Path,
    split_path: Path,
    calibrate_container: Path,
    flagger_container: Path,
    expected_ms: int = 36,
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
        expected_ms (int): Expected numbner of measurement sets that should reside in the ``bandpass_path``
        calibrate_container (Path): Path to a singularity container with the ao-calibrate tool
        flagger_container (Path): Path to a singularity container with aoflaffer

    Returns:
        Path: Directory that contains the extracted measurement sets and the ao-style gain solutions files.
    """
    assert (
        bandpass_path.exists() and bandpass_path.is_dir()
    ), f"{str(bandpass_path)} does not exist or is not a folder. "
    bandpass_mss = list([MS.cast(ms_path) for ms_path in bandpass_path.glob("*.ms")])

    assert (
        len(bandpass_mss) == expected_ms
    ), f"Expected to find {expected_ms} in {str(bandpass_path)}, found {len(bandpass_mss)}."

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
        calibrate_container=calibrate_container,
        flagger_container=flagger_container,
        model_path=model_path,
        source_name_prefix=source_name_prefix,
    )

    return output_split_bandpass_path


def setup_run_bandpass_flow(
    bandpass_path: Path,
    split_path: Path,
    expected_ms: int,
    calibrate_container: Path,
    flagger_container: Path,
    cluster_config: Path,
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
        expected_ms (int): Expected numbner of measurement sets that should reside in the ``bandpass_path``
        calibrate_container (Path): Path to a singularity container with the ao-calibrate tool
        flagger_container (Path): Path to a singularity container with aoflaffer
        cluster_config (Path): Path to a yaml file that is used to configure a prefect dask task runner.

    Returns:
        Path: Directory that contains the extracted measurement sets and the ao-style gain solutions files.
    """

    dask_task_runner = get_dask_runner(cluster=cluster_config)

    bandpass_sbid = get_sbid_from_path(path=bandpass_path)

    calibrate_bandpass_flow.with_options(
        f"Flint Bandpass Pipeline -- {bandpass_sbid}", task_runner=dask_task_runner
    )(
        bandpass_path=bandpass_path,
        split_path=split_path,
        calibrate_container=calibrate_container,
        flagger_container=flagger_container,
        expected_ms=expected_ms,
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

    return parser


def cli() -> None:
    import logging

    logger.setLevel(logging.INFO)

    parser = get_parser()

    args = parser.parse_args()

    setup_run_bandpass_flow(
        bandpass_path=args.bandopass_path,
        split_path=args.split_path,
        expected_ms=args.expected_ms,
        calibrate_container=args.calibrate_container,
        flagger_container=args.flagger_container,
        cluster_config=args.cluster_config,
    )


if __name__ == "__main__":
    cli()
