from __future__ import annotations

from pathlib import Path
from typing import Any

from prefect import flow, tags, unmapped

from flint.configuration import (
    get_options_from_strategy,
    load_and_copy_strategy,
)
from flint.exceptions import MSError
from flint.logging import logger
from flint.ms import find_mss
from flint.naming import (
    ProcessedNameComponents,
    add_timestamp_to_path,
    extract_components_from_name,
)
from flint.options import (
    PolFieldOptions,
    dump_field_options_to_yaml,
)
from flint.prefect.common.imaging import (
    create_convol_linmos_images,
    create_convolve_linmos_cubes,
    task_wsclean_imager,
    task_zip_ms,
)
from flint.prefect.common.utils import (
    task_archive_sbid,
)


@flow(name="Flint Polarisation Pipeline")
def process_science_fields_pol(
    flint_ms_directory: Path,
    pol_field_options: PolFieldOptions,
) -> None:
    # Verify no nasty incompatible options
    # _check_field_options(field_options=field_options)

    # Get some placeholder names
    science_mss = find_mss(
        mss_parent_path=flint_ms_directory,
        expected_ms_count=pol_field_options.expected_ms,
    )

    dump_field_options_to_yaml(
        output_path=add_timestamp_to_path(
            input_path=flint_ms_directory / "pol_field_options.yaml"
        ),
        field_options=pol_field_options,
    )

    archive_wait_for: list[Any] = []

    strategy = load_and_copy_strategy(
        output_split_science_path=flint_ms_directory,
        imaging_strategy=pol_field_options.imaging_strategy,
    )

    logger.info(f"{pol_field_options=}")

    components = extract_components_from_name(science_mss[0].path)

    if not isinstance(components, ProcessedNameComponents):
        msg = f"{science_mss[0]} has not be processed by Flint"
        raise MSError(msg)

    logger.info(f"Found the following calibrated measurement sets: {science_mss}")

    if pol_field_options.wsclean_container is None:
        logger.info("No wsclean container provided. Returning. ")
        return

    wsclean_cmds = task_wsclean_imager.map(
        in_ms=science_mss,
        wsclean_container=pol_field_options.wsclean_container,
        strategy=unmapped(strategy),
        mode="wsclean",
    )  # type: ignore

    # TODO: This should be waited!
    archive_wait_for.extend(wsclean_cmds)

    if pol_field_options.yandasoft_container:
        parsets = create_convol_linmos_images(
            wsclean_cmds=wsclean_cmds,
            field_options=pol_field_options,
            field_summary=None,
            current_round=None,
        )
        archive_wait_for.extend(parsets)

    # Always create cubes
    with tags("cubes"):
        cube_parset = create_convolve_linmos_cubes(
            wsclean_cmds=wsclean_cmds,  # type: ignore
            field_options=pol_field_options,
            current_round=None,
            additional_linmos_suffix_str="cube",
        )
        archive_wait_for.append(cube_parset)

    # zip up the final measurement set, which is not included in the above loop
    if pol_field_options.zip_ms:
        archive_wait_for = task_zip_ms.map(
            in_item=wsclean_cmds, wait_for=archive_wait_for
        )

    if pol_field_options.sbid_archive_path or pol_field_options.sbid_copy_path:
        update_archive_options = get_options_from_strategy(
            strategy=strategy, mode="archive"
        )
        task_archive_sbid.submit(
            science_folder_path=flint_ms_directory,
            archive_path=pol_field_options.sbid_archive_path,
            copy_path=pol_field_options.sbid_copy_path,
            max_round=None,
            update_archive_options=update_archive_options,
            wait_for=archive_wait_for,
        )  # type: ignore
