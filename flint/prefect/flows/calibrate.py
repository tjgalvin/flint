"""A prefect based pipeline to bandpass calibrate data and apply it to a science observation
"""
import logging
from pathlib import Path
from argparse import ArgumentParser
from typing import Union, Any, List

from prefect import task, flow

from flint.logging import logger
from flint.ms import MS, preprocess_askap_ms, split_by_field
from flint.bandpass import extract_correct_bandpass_pointing, plot_solutions
from flint.flagging import flag_ms_aoflagger
from flint.calibrate.aocalibrate import (
    create_calibrate_cmd,
    CalibrateCommand,
    select_aosolution_for_ms,
    create_apply_solutions_cmd,
)
from flint.sky_model import get_1934_model
from flint.prefect.clusters import get_dask_runner

task_extract_correct_bandpass_pointing = task(extract_correct_bandpass_pointing)
task_preprocess_askap_ms = task(preprocess_askap_ms)
task_flag_ms_aoflagger = task(flag_ms_aoflagger)
task_create_calibrate_cmd = task(create_calibrate_cmd)
task_split_by_field = task(split_by_field)
task_select_solution_for_ms = task(select_aosolution_for_ms)
task_create_apply_solutions_cmd = task(create_apply_solutions_cmd)


@task
def task_flatten_prefect_futures(in_futures: List[List[Any]]) -> List[Any]:
    """Will flatten a list of lists into a single list. This
    is useful for when a task-descorated function returns a list.


    Args:
        in_futures (List[List[Any]]): Input list of lists to flatten

    Returns:
        List[Any]: Flattened form of input
    """
    logger.info(f"Received {len(in_futures)} to flatten.")
    flatten_list = [item for sublist in in_futures for item in sublist]
    logger.info(f"Flattened list {len(flatten_list)}")

    return flatten_list


@task
def task_plot_solutions(calibrate_cmd: CalibrateCommand) -> None:
    plot_solutions(solutions_path=calibrate_cmd.solution_path, ref_ant=0)


@flow(name="Bandpass")
def process_bandpass_science_fields(
    bandpass_path: Path,
    science_path: Path,
    split_path: Path,
    flagger_container: Path,
    calibrate_container: Path,
    expected_ms: int = 36,
    source_name_prefix: str = "B1934-638",
) -> None:
    assert (
        bandpass_path.exists() and bandpass_path.is_dir()
    ), f"{str(bandpass_path)} does not exist or is not a folder. "
    bandpass_mss = list([MS.cast(ms_path) for ms_path in bandpass_path.glob(f"*.ms")])
    assert (
        len(bandpass_mss) == expected_ms
    ), f"Expected to find {expected_ms} in {str(bandpass_path)}, found {len(bandpass_mss)}."

    assert (
        science_path.exists() and science_path.is_dir()
    ), f"{str(science_path)} does not exist or is not a folder. "
    science_mss = list([MS.cast(ms_path) for ms_path in science_path.glob(f"*.ms")])
    assert (
        len(science_mss) == expected_ms
    ), f"Expected to find {expected_ms} in {str(science_path)}, found {len(bandpass_mss)}."

    logger.info(
        f"Found the following bandpass measurement set: {[bp.path for bp in bandpass_mss]}."
    )
    bandpass_folder_name = bandpass_path.name
    science_folder_name = science_path.name

    output_split_bandpass_path = (
        Path(split_path / bandpass_folder_name).absolute().resolve()
    )
    output_split_science_path = (
        Path(split_path / science_folder_name).absolute().resolve()
    )
    logger.info(
        f"Will write extracted bandpass MSs to: {str(output_split_bandpass_path)}."
    )
    if not output_split_bandpass_path.exists():
        logger.info(f"Creating {str(output_split_bandpass_path)}")
        output_split_bandpass_path.mkdir(parents=True)

    if not output_split_science_path.exists():
        logger.info(f"Creating {str(output_split_science_path)}")
        output_split_science_path.mkdir(parents=True)

    model_path: Path = get_1934_model(mode="calibrate")
    calibrate_cmds: List[CalibrateCommand] = []

    for bandpass_ms in bandpass_mss:
        extract_bandpass_ms = task_extract_correct_bandpass_pointing.submit(
            ms=bandpass_ms,
            source_name_prefix=source_name_prefix,
            ms_out_dir=output_split_bandpass_path,
        )
        preprocess_bandpass_ms = task_preprocess_askap_ms.submit(ms=extract_bandpass_ms)
        flag_bandpass_ms = task_flag_ms_aoflagger.submit(
            ms=preprocess_bandpass_ms, container=flagger_container, rounds=3
        )
        calibrate_cmd = task_create_calibrate_cmd.submit(
            ms=flag_bandpass_ms,
            calibrate_model=model_path,
            container=calibrate_container,
        )
        task_plot_solutions.submit(calibrate_cmd=calibrate_cmd)
        calibrate_cmds.append(calibrate_cmd)

    science_fields = []
    for science_ms in science_mss:
        split_science_ms = task_split_by_field.submit(
            ms=science_ms, field=None, out_dir=output_split_science_path
        )
        science_fields.append(split_science_ms)

    # The following line will block until the science
    # fields are split out. Since there might be more
    # than a single field in an SBID, we should do this
    field_science_mss = task_flatten_prefect_futures(science_fields)

    for field_science_ms in field_science_mss:
        logger.info(f"Processing {field_science_ms}.")
        preprocess_science_ms = task_preprocess_askap_ms.submit(
            ms=field_science_ms,
            data_column="DATA",
            instrument_column="INSTRUMENT_DATA",
            overwrite=True,
        )
        flag_field_ms = task_flag_ms_aoflagger.submit(
            ms=preprocess_science_ms, container=flagger_container, rounds=3
        )

        solutions_path = task_select_solution_for_ms.submit(
            calibrate_cmds=calibrate_cmds, ms=flag_field_ms, wait_for=calibrate_cmds
        )
        apply_solutions_cmd = task_create_apply_solutions_cmd.submit(
            ms=flag_field_ms,
            solutions_file=solutions_path,
            container=calibrate_container,
        )


def setup_run_process_science_field(
    cluster_config: Union[str, Path],
    bandpass_path: Path,
    science_path: Path,
    split_path: Path,
    flagger_container: Path,
    calibrate_container: Path,
    expected_ms: int = 36,
    source_name_prefix: str = "B1934-638",
) -> None:
    dask_task_runner = get_dask_runner(cluster=cluster_config)

    process_bandpass_science_fields.with_options(
        name="Flint Bandpass", task_runner=dask_task_runner
    )(
        bandpass_path=bandpass_path,
        science_path=science_path,
        split_path=split_path,
        flagger_container=flagger_container,
        calibrate_container=calibrate_container,
        expected_ms=expected_ms,
        source_name_prefix=source_name_prefix,
    )


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)

    parser.add_argument(
        "bandpass_path",
        type=Path,
        help="Path to directory containing the beam-wise measurement sets that contain the bandpass calibration source. ",
    )

    parser.add_argument(
        "science_path",
        type=Path,
        help="Path to directories containing the beam-wise science measurementsets that will have solutions copied over and applied.",
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        default=Path("."),
        help="Location to write field-split MSs to. Will attempt to use the parent name of a directory when writing out a new MS. ",
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

    # logger = logging.getLogger("flint")
    logger.setLevel(logging.INFO)

    parser = get_parser()

    args = parser.parse_args()

    setup_run_process_science_field(
        cluster_config=args.cluster_config,
        bandpass_path=args.bandpass_path,
        science_path=args.science_path,
        split_path=args.split_path,
        flagger_container=args.flagger_container,
        calibrate_container=args.calibrate_container,
        expected_ms=args.expected_ms,
    )


if __name__ == "__main__":
    cli()
