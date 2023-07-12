"""A prefect based pipeline to bandpass calibrate data and apply it to a science observation
"""

from pathlib import Path
from argparse import ArgumentParser
from typing import Optional, Union

from prefect import task, flow, get_run_logger

from flint.logging import logger
from flint.ms import MS, preprocess_askap_ms
from flint.bandpass import extract_correct_bandpass_pointing
from flint.flagging import flag_ms_aoflagger
from flint.calibrate.aocalibrate import create_calibrate_cmd
from flint.sky_model import get_1934_model
from flint.prefect.clusters import get_dask_runner

task_extract_correct_bandpass_pointing = task(extract_correct_bandpass_pointing)
task_preprocess_askap_ms = task(preprocess_askap_ms)
task_flag_ms_aoflagger = task(flag_ms_aoflagger)
task_create_calibrate_cmd = task(create_calibrate_cmd)


@flow(name="Bandpass")
def process_bandpass_science_fields(
    bandpass_path: Path,
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
    if not output_split_bandpass_path.exists():
        logger.info(f"Creating {str(output_split_bandpass_path)}")
        output_split_bandpass_path.mkdir(parents=True)

    model_path: Path = get_1934_model(mode="calibrate")

    for bandpass_ms in bandpass_mss:
        extract_bandpass_ms = task_extract_correct_bandpass_pointing.submit(
            ms=bandpass_ms,
            source_name_prefix=source_name_prefix,
            ms_out_dir=output_split_bandpass_path,
        )
        preprocess_bandpass_ms = task_preprocess_askap_ms.submit(ms=extract_bandpass_ms)
        flag_bandpass_ms = task_flag_ms_aoflagger.submit(
            ms=preprocess_bandpass_ms, container=flagger_container
        )
        calibrate_cmd = task_create_calibrate_cmd.submit(
            ms=flag_bandpass_ms,
            calibrate_model=model_path,
            container=calibrate_container,
        )


def setup_run_process_science_field(
    cluster_config: Union[str, Path],
    bandpass_path: Path,
    split_path: Path,
    flagger_container: Path,
    calibrate_container: Path,
    expected_ms: int = 36,
    source_name_prefix: str = "B1934-638",
) -> None:
    dask_task_runner = get_dask_runner(cluster=cluster_config)

    process_bandpass_science_fields.with_options(
        name="Test_Bandpass", task_runner=dask_task_runner
    )(
        bandpass_path=bandpass_path,
        split_path=split_path,
        flagger_container=flagger_container,
        calibrate_container=calibrate_container,
        expected_ms=expected_ms,
        source_name_prefix=source_name_prefix,
    )

    pass


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)

    parser.add_argument(
        "bandpass_path",
        type=Path,
        help="Path to directory containing the beam-wise measurement sets that contain the bandpass calibration source. ",
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

    logger.setLevel(logging.INFO)

    parser = get_parser()

    args = parser.parse_args()

    setup_run_process_science_field(
        cluster_config=args.cluster_config,
        bandpass_path=args.bandpass_path,
        split_path=args.split_path,
        flagger_container=args.flagger_container,
        calibrate_container=args.calibrate_container,
        expected_ms=args.expected_ms,
    )


if __name__ == "__main__":
    cli()
