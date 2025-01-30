from __future__ import annotations

from pathlib import Path

from configargparse import ArgumentParser
from prefect import flow, tags
from prefect.futures import PrefectFuture

from flint.coadd.linmos import LinmosResult
from flint.configuration import (
    POLARISATION_MAPPING,
    get_options_from_strategy,
    load_and_copy_strategy,
)
from flint.convol import task_convolve_images
from flint.exceptions import MSError
from flint.imager.wsclean import (
    ImageSet,
    WSCleanResult,
    combine_images_to_cube,
    task_merge_image_sets,
    task_split_and_get_image_set,
)
from flint.logging import logger
from flint.ms import find_mss
from flint.naming import (
    ProcessedNameComponents,
    add_timestamp_to_path,
    extract_components_from_name,
    get_sbid_from_path,
)
from flint.options import (
    PolFieldOptions,
    add_options_to_parser,
    create_options_from_parser,
    dump_field_options_to_yaml,
)
from flint.prefect.clusters import get_dask_runner
from flint.prefect.common.imaging import (
    task_get_common_beam_from_image_set,
    task_linmos_images,
    task_wsclean_imager,
)
from flint.prefect.common.utils import (
    task_create_field_summary,
    task_get_channel_images_from_paths,
    task_getattr,
    task_rename_linear_to_stokes,
)


@flow(name="Flint Polarisation Pipeline")
def process_science_fields_pol(
    flint_ms_directory: Path,
    pol_field_options: PolFieldOptions,
) -> None:
    strategy = load_and_copy_strategy(
        output_split_science_path=flint_ms_directory,
        imaging_strategy=pol_field_options.imaging_strategy,
    )

    logger.info(f"{pol_field_options=}")

    if strategy is None:
        logger.info("No strategy provided. Returning.")
        return

    # Get some placeholder names
    science_mss = find_mss(
        mss_parent_path=flint_ms_directory,
        expected_ms_count=pol_field_options.expected_ms,
        data_column=strategy["defaults"].get("data_column", "DATA"),
    )

    field_summary = task_create_field_summary.submit(
        mss=science_mss,
        holography_path=pol_field_options.holofile,
    )

    dump_field_options_to_yaml(
        output_path=add_timestamp_to_path(
            input_path=flint_ms_directory / "pol_field_options.yaml"
        ),
        field_options=pol_field_options,
    )

    # Check that the MSs have been processed by Flint
    for ms in science_mss:
        components = extract_components_from_name(ms.path)
        if not isinstance(components, ProcessedNameComponents):
            msg = f"{ms} has not be processed by Flint"
            raise MSError(msg)

    logger.info(f"Found the following calibrated measurement sets: {science_mss}")

    if pol_field_options.wsclean_container is None:
        logger.info("No wsclean container provided. Returning. ")
        return

    polarisations: dict[str, str] = strategy.get("polarisation", {"total": {}})

    image_sets_dict: dict[str, PrefectFuture[ImageSet]] = {}
    image_sets_list: list[PrefectFuture[ImageSet]] = []
    for polarisation in polarisations.keys():
        _image_sets = []
        with tags(f"polarisation-{polarisation}"):
            for science_ms in science_mss:
                wsclean_result: PrefectFuture[WSCleanResult] = (
                    task_wsclean_imager.submit(
                        in_ms=science_ms,
                        wsclean_container=pol_field_options.wsclean_container,
                        make_cube_from_subbands=False,  # We will do this later
                        update_wsclean_options=get_options_from_strategy(
                            strategy=strategy,
                            operation="polarisation",
                            mode="wsclean",
                            polarisation=polarisation,
                        ),
                    )
                )
                _image_set: PrefectFuture[ImageSet] = task_getattr.submit(
                    wsclean_result, "image_set"
                )
                _image_sets.append(_image_set)
                image_sets_list.append(_image_set)
        image_sets_dict[polarisation] = _image_sets

    merged_image_set = task_merge_image_sets.submit(image_sets=image_sets_list)

    common_beam_shape = task_get_common_beam_from_image_set.submit(
        image_set=merged_image_set,
        cutoff=pol_field_options.beam_cutoff,
        fixed_beam_shape=pol_field_options.fixed_beam_shape,
    )

    stokes_beam_cubes: dict[str, list[PrefectFuture[Path]]] = {}
    for polarisation, image_set_list in image_sets_dict.items():
        with tags(f"polarisation-{polarisation}"):
            # Get the individual Stokes parameters in case of joint imaging
            if polarisation not in POLARISATION_MAPPING.keys():
                raise ValueError(f"Unknown polarisation {polarisation}")
            stokes_list = list(POLARISATION_MAPPING[polarisation])
            for stokes in stokes_list:
                with tags(f"stokes-{stokes}"):
                    beam_cubes: list[PrefectFuture[Path]] = []
                    for image_set in image_set_list:
                        stokes_image_list = task_split_and_get_image_set.submit(
                            image_set=image_set,
                            get=stokes,
                            by="pol",
                            mode="image",
                        )
                        convolved_image_list = task_convolve_images.submit(
                            image_paths=stokes_image_list,
                            beam_shape=common_beam_shape,
                            cutoff=pol_field_options.beam_cutoff,
                        )
                        # TODO: Consider accerating this by doing a linmos per-channel, then combining
                        channel_image_list = task_get_channel_images_from_paths.submit(
                            paths=convolved_image_list
                        )
                        prefix = task_getattr.submit(image_set, "prefix")

                        if polarisation == "linear":
                            # Get single Stokes prefix - the original prefix is the linear prefix
                            # i.e. `.qu.` -> `.q.` or `.u.` depending on the stokes
                            prefix = task_rename_linear_to_stokes.submit(
                                linear_name=prefix,
                                stokes=stokes,
                            )
                        cube_path = combine_images_to_cube.submit(
                            images=channel_image_list,
                            prefix=prefix,
                            mode="image",
                            remove_original_images=True,
                        )
                        beam_cubes.append(cube_path)
                stokes_beam_cubes[stokes] = beam_cubes

    linmos_result_list: list[PrefectFuture[LinmosResult]] = []
    # We run linmos now to ensure we have Stokes I images for leakage correction
    # If we have not imaged Stokes I, we cannot do leakage correction
    force_remove_leakage: bool | None = None
    if "i" not in stokes_beam_cubes.keys():
        force_remove_leakage = False

    for stokes, beam_cubes in stokes_beam_cubes.items():
        with tags(f"stokes-{stokes}"):
            linmos_result = task_linmos_images.submit(
                images=beam_cubes,
                container=pol_field_options.yandasoft_container,
                holofile=pol_field_options.holofile,
                cutoff=pol_field_options.pb_cutoff,
                field_summary=field_summary,
                stokesi_images=stokes_beam_cubes.get("i"),
                force_remove_leakage=force_remove_leakage,
                trim_linmos_fits=pol_field_options.trim_linmos_fits,
            )
            linmos_result_list.append(linmos_result)

    # wait for all linmos results to be completed
    _ = [linmos_result.result() for linmos_result in linmos_result_list]


def setup_run_process_science_field(
    cluster_config: str | Path,
    flint_ms_directory: Path,
    pol_field_options: PolFieldOptions,
) -> None:
    science_sbid = get_sbid_from_path(path=flint_ms_directory)

    if pol_field_options.sbid_copy_path:
        updated_sbid_copy_path = pol_field_options.sbid_copy_path / f"{science_sbid}"
        logger.info(f"Updating archive copy path to {updated_sbid_copy_path=}")
        pol_field_options = pol_field_options.with_options(
            sbid_copy_path=updated_sbid_copy_path
        )

    dask_task_runner = get_dask_runner(cluster=cluster_config)

    process_science_fields_pol.with_options(
        name=f"Flint Polarisation Pipeline - {science_sbid}",
        task_runner=dask_task_runner,
    )(
        flint_ms_directory=flint_ms_directory,
        pol_field_options=pol_field_options,
    )


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)

    parser.add_argument(
        "--cli-config", is_config_file=True, help="Path to configuration file"
    )

    parser.add_argument(
        "flint_ms_directory",
        type=Path,
        help="Path to directories containing the beam-wise flint-calibrated MeasurementSets.",
    )
    parser.add_argument(
        "--cluster-config",
        type=str,
        default="petrichor",
        help="Path to a cluster configuration file, or a known cluster name. ",
    )

    parser = add_options_to_parser(
        parser=parser,
        options_class=PolFieldOptions,
        description="Polarisation processing options",
    )

    return parser


def cli() -> None:
    import logging

    # logger = logging.getLogger("flint")
    logger.setLevel(logging.INFO)

    parser = get_parser()

    args = parser.parse_args()

    field_options = create_options_from_parser(
        parser_namespace=args,
        options_class=PolFieldOptions,
    )

    setup_run_process_science_field(
        cluster_config=args.cluster_config,
        flint_ms_directory=args.flint_ms_directory,
        pol_field_options=field_options,
    )


if __name__ == "__main__":
    cli()
