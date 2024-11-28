"""This is a workflow to subtract a continuum model and image the channel-wise data

Unlike the continuum imaging and self-calibnration pipeline this flow currently
expects that all measurement sets are in the flint format, which means other than
the naming scheme that they have been been preprocessed to place them in the IAU
frame and have had their fields table updated. That is to say that they have
already been preprocessed and fixed.
"""

from pathlib import Path
from typing import Tuple, Optional, Any

import numpy as np
from configargparse import ArgumentParser
from prefect import flow, unmapped, task

from flint.configuration import _load_and_copy_strategy
from flint.exceptions import FrequencyMismatchError
from flint.prefect.clusters import get_dask_runner
from flint.logging import logger
from flint.ms import MS, find_mss, consistent_ms_frequencies, get_freqs_from_ms
from flint.options import (
    SubtractFieldOptions,
    add_options_to_parser,
    create_options_from_parser,
)
from flint.prefect.common.imaging import (
    task_wsclean_imager,
    task_get_common_beam,
    _convolve_linmos,
)
from flint.naming import get_sbid_from_path


@task
def task_gather_results(*args_of_futures: Any) -> Any:
    return args_of_futures


def _check_and_verify_options(subtract_field_options: SubtractFieldOptions) -> None:
    """Verrify that the options supplied to run the subtract field options make sense"""
    assert (
        subtract_field_options.calibrate_container.exists()
        and subtract_field_options.calibrate_container.is_file()
    ), f"{subtract_field_options.calibrate_container=} does not exist or is not a file"
    assert (
        subtract_field_options.wsclean_container.exists()
        and subtract_field_options.wsclean_container.is_file()
    ), f"{subtract_field_options.wsclean_container=} does not exist or is not a file"
    assert (
        subtract_field_options.yandasoft_container.exists()
        and subtract_field_options.yandasoft_container.is_file()
    ), f"{subtract_field_options.yandasoft_container=} does not exist or is not a file"


def find_mss_to_image(
    mss_parent_path: Path,
    expected_ms_count: Optional[int] = None,
    data_column: str = "CORRECTED_DATA",
) -> Tuple[MS, ...]:
    science_mss = find_mss(
        mss_parent_path=mss_parent_path,
        expected_ms_count=expected_ms_count,
        data_column=data_column,
    )
    logger.info(f"Found {science_mss=}")
    return science_mss


@flow
def flow_subtract_cube(
    science_path: Path, subtract_field_options: SubtractFieldOptions
) -> None:
    strategy = _load_and_copy_strategy(
        output_split_science_path=science_path,
        imaging_strategy=subtract_field_options.imaging_strategy,
    )
    _check_and_verify_options(subtract_field_options=subtract_field_options)

    # Find the MSs
    # - optionally untar?
    science_mss = find_mss_to_image(
        mss_parent_path=science_path,
        expected_ms_count=subtract_field_options.expected_ms,
        data_column=subtract_field_options.data_column,
    )

    # 2 - ensure matchfing frequencies over channels
    consistent_frequencies_across_mss = consistent_ms_frequencies(mss=science_mss)
    if not consistent_frequencies_across_mss:
        logger.critical("Mismatch in frequencies among provided MSs")
        raise FrequencyMismatchError("There is a mismatch in frequencies")

    # 2.5 - Continuum subtract if requested

    freqs_mhz = get_freqs_from_ms(ms=science_mss[0]) / 1e6
    logger.info(
        f"Considering {len(freqs_mhz)} from {len(science_mss)}, minimum {np.min(freqs_mhz)}-{np.max(freqs_mhz)}"
    )

    # 3 - out loop over channels to image
    #   a - wsclean map over the ms and channels
    #   b - convol to a common resolution for channels
    #   c - linmos the smoothed images together

    channel_parset_list = []
    batched_channel_parset_list = []
    for channel, freq_mhz in enumerate(freqs_mhz):
        logger.info(f"Imaging {channel=} {freq_mhz=}")
        channel_range = (channel, channel + 1)
        channel_wsclean_cmds = task_wsclean_imager.map(
            in_ms=science_mss,
            wsclean_container=subtract_field_options.wsclean_container,
            channel_range=unmapped(channel_range),
            strategy=unmapped(strategy),
            mode="wsclean",
            operation="subtractcube",
        )
        channel_beam_shape = task_get_common_beam.submit(
            wsclean_cmds=channel_wsclean_cmds,
            cutoff=subtract_field_options.beam_cutoff,
            filter="image.",
        )
        channel_parset = _convolve_linmos(
            wsclean_cmds=channel_wsclean_cmds,
            beam_shape=channel_beam_shape,
            linmos_suffix_str=f"ch{channel_range[0]}-{channel_range[1]}",
            field_options=subtract_field_options,
            convol_mode="image",
            convol_filter="image.",
            convol_suffix_str="optimal.image",
            trim_linmos_fits=False,
            remove_original_images=True,
        )
        batched_channel_parset_list.append(channel_parset)

        if len(batched_channel_parset_list) >= subtract_field_options.batch_limit:
            logger.info("Popping a result")
            from prefect.states import Completed

            found_a_result = False
            while not found_a_result:
                # Attempt to find a result already completed
                for idx, future in enumerate(batched_channel_parset_list):
                    if future.get_state() == Completed:
                        channel_parset_list.append(batched_channel_parset_list.pop(idx))
                        found_a_result = True
                        break
    else:
        channel_parset_list.extend(
            [parset.result() for parset in batched_channel_parset_list]
        )

    # 4 - cube concatenated each linmos field together to single file

    return


def setup_run_subtract_flow(
    science_path: Path,
    subtract_field_options: SubtractFieldOptions,
    cluster_config: Path,
) -> None:
    logger.info(f"Processing {science_path=}")
    science_sbid = get_sbid_from_path(path=science_path)

    dask_runner = get_dask_runner(cluster=cluster_config)

    flow_subtract_cube.with_options(
        task_runner=dask_runner, name=f"Subtract Cube Pipeline -- {science_sbid}"
    )(science_path=science_path, subtract_field_options=subtract_field_options)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cli-config", is_config_file=True, help="Path to configuration file"
    )
    parser.add_argument(
        "science_path",
        type=Path,
        help="Path to the directory containing the beam-wise measurement sets",
    )
    parser.add_argument(
        "--cluster-config",
        type=str,
        default="petrichor",
        help="Path to a cluster configuration file, or a known cluster name. ",
    )

    parser = add_options_to_parser(parser=parser, options_class=SubtractFieldOptions)

    return parser


def cli() -> None:
    parser = get_parser()

    args = parser.parse_args()

    subtract_field_options = create_options_from_parser(
        parser_namespace=args, options_class=SubtractFieldOptions
    )

    setup_run_subtract_flow(
        science_path=args.science_path,
        subtract_field_options=subtract_field_options,
        cluster_config=args.cluster_config,
    )


if __name__ == "__main__":
    cli()
