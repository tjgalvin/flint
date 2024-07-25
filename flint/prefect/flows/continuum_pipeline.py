"""A prefect based pipeline that:
- will perform bandpass calibration with PKS B1934-638 data, or from a derived sky-model
- copy and apply to science field
- image and self-calibration the science fields
- run aegean source finding
"""

from asyncio import wait_for
from pathlib import Path
from typing import Any, List, Optional, Union

from configargparse import ArgumentParser
from prefect import flow, tags, unmapped

from flint.calibrate.aocalibrate import find_existing_solutions
from flint.configuration import (
    Strategy,
    copy_and_timestamp_strategy_file,
    get_options_from_strategy,
    load_strategy_yaml,
)
from flint.logging import logger
from flint.masking import consider_beam_mask_round
from flint.ms import find_mss
from flint.naming import (
    CASDANameComponents,
    add_timestamp_to_path,
    extract_components_from_name,
    get_sbid_from_path,
)
from flint.options import FieldOptions, dump_field_options_to_yaml
from flint.prefect.clusters import get_dask_runner
from flint.prefect.common.imaging import (
    _create_convol_linmos_images,
    _validation_items,
    task_copy_and_preprocess_casda_askap_ms,
    task_create_apply_solutions_cmd,
    task_create_image_mask_model,
    task_flag_ms_aoflagger,
    task_gaincal_applycal_ms,
    task_potato_peel,
    task_preprocess_askap_ms,
    task_rename_column_in_ms,
    task_run_bane_and_aegean,
    task_select_solution_for_ms,
    task_split_by_field,
    task_wsclean_imager,
    task_zip_ms,
)
from flint.prefect.common.utils import (
    task_archive_sbid,
    task_create_beam_summary,
    task_create_field_summary,
    task_flatten,
    task_update_field_summary,
    task_update_with_options,
)
from flint.selfcal.utils import consider_skip_selfcal_on_round


def _check_field_options(field_options: FieldOptions) -> None:
    run_aegean = (
        False if field_options.aegean_container is None else field_options.run_aegean
    )
    if field_options.use_beam_masks is True and run_aegean is False:
        raise ValueError(
            "run_aegean and aegean container both need to be set is beam masks is being used. "
        )


def _check_create_output_split_science_path(
    science_path: Path, split_path: Path, check_exists: bool = True
) -> Path:
    """Create the output path that the science MSs will be placed.

    Args:
        science_path (Path): The directory that contains the MSs for science processing
        split_path (Path): Where the output MSs will be written to and processed
        check_exists (bool, optional): Should we check to make sure output directory does not exist. Defaults to True.

    Raises:
        ValueError: Raised when the output directory exists

    Returns:
        Path: The output directory
    """

    science_folder_name = science_path.name
    assert str(
        science_folder_name
    ).isdigit(), f"We require the parent directory to be the SBID (all digits), got {science_folder_name=}"
    output_split_science_path = (
        Path(split_path / science_folder_name).absolute().resolve()
    )

    if check_exists and output_split_science_path.exists():
        logger.critical(
            f"{output_split_science_path=} already exists. It should not. Exiting. "
        )
        raise ValueError("Output science directory already exists. ")

    logger.info(f"Creating {str(output_split_science_path)}")
    output_split_science_path.mkdir(parents=True)

    return output_split_science_path


def _load_and_copy_strategy(
    output_split_science_path: Path, imaging_strategy: Optional[Path] = None
) -> Union[Strategy, None]:
    """Load a strategy file and copy a timestamped version into the output directory
    that would contain the science processing.

    Args:
        output_split_science_path (Path): Where the strategy file should be copied to (where the data would be processed)
        imaging_strategy (Optional[Path], optional): Location of the strategy file. Defaults to None.

    Returns:
        Union[Strategy, None]: The loadded strategy file if provided, `None` otherwise
    """
    return (
        load_strategy_yaml(
            input_yaml=copy_and_timestamp_strategy_file(
                output_dir=output_split_science_path,
                input_yaml=imaging_strategy,
            ),
            verify=True,
        )
        if imaging_strategy
        else None
    )


@flow(name="Flint Continuum Pipeline")
def process_science_fields(
    science_path: Path,
    split_path: Path,
    field_options: FieldOptions,
    bandpass_path: Optional[Path] = None,
) -> None:
    # Verify no nasty incompatible options
    _check_field_options(field_options=field_options)

    # Get some placeholder names
    run_aegean = (
        False if field_options.aegean_container is None else field_options.run_aegean
    )  # This is also in check_field_options
    run_validation = field_options.reference_catalogue_directory is not None

    science_mss = find_mss(
        mss_parent_path=science_path, expected_ms_count=field_options.expected_ms
    )

    output_split_science_path = _check_create_output_split_science_path(
        science_path=science_path, split_path=split_path, check_exists=True
    )

    dump_field_options_to_yaml(
        output_path=add_timestamp_to_path(
            input_path=output_split_science_path / "field_options.yaml"
        ),
        field_options=field_options,
    )

    archive_wait_for: List[Any] = []

    strategy = _load_and_copy_strategy(
        output_split_science_path=output_split_science_path,
        imaging_strategy=field_options.imaging_strategy,
    )

    logger.info(f"{field_options=}")

    logger.info(f"Found the following raw measurement sets: {science_mss}")

    # TODO: This feels a little too much like that feeling of being out
    # at sea for to long. Should refactor (or mask a EMU only).
    if isinstance(
        extract_components_from_name(name=science_mss[0].path), CASDANameComponents
    ):
        preprocess_science_mss = task_copy_and_preprocess_casda_askap_ms.map(
            casda_ms=science_mss, output_directory=output_split_science_path
        )
    else:
        # TODO: This will likely need to be expanded should any
        # other calibration strategies get added
        # Scan the existing bandpass directory for the existing solutions
        assert bandpass_path, f"{bandpass_path=}, it needs to be set"

        calibrate_cmds = find_existing_solutions(
            bandpass_directory=bandpass_path,
            use_preflagged=field_options.use_preflagger,
            use_smoothed=field_options.use_smoothed,
        )

        logger.info(f"Constructed the following {calibrate_cmds=}")

        split_science_mss = task_split_by_field.map(
            ms=science_mss,
            field=None,
            out_dir=unmapped(output_split_science_path),
            column=unmapped("DATA"),
        )

        # This will block until resolved
        flat_science_mss = task_flatten.submit(split_science_mss).result()

        solutions_paths = task_select_solution_for_ms.map(
            calibrate_cmds=unmapped(calibrate_cmds), ms=flat_science_mss
        )
        apply_solutions_cmds = task_create_apply_solutions_cmd.map(
            ms=flat_science_mss,
            solutions_file=solutions_paths,
            container=field_options.calibrate_container,
        )
        flagged_mss = task_flag_ms_aoflagger.map(
            ms=apply_solutions_cmds, container=field_options.flagger_container
        )
        column_rename_mss = task_rename_column_in_ms.map(
            ms=flagged_mss,
            original_column_name=unmapped("DATA"),
            new_column_name=unmapped("INSTRUMENT_DATA"),
        )
        preprocess_science_mss = task_preprocess_askap_ms.map(
            ms=column_rename_mss,
            data_column=unmapped("CORRECTED_DATA"),
            instrument_column=unmapped("DATA"),
            overwrite=True,
        )

    if field_options.no_imaging:
        logger.info(
            f"No imaging will be performed, as requested bu {field_options.no_imaging=}"
        )
        return

    field_summary = task_create_field_summary.submit(
        mss=preprocess_science_mss,
        cal_sbid_path=bandpass_path,
        holography_path=field_options.holofile,
    )
    logger.info(f"{field_summary=}")

    if field_options.wsclean_container is None:
        logger.info("No wsclean container provided. Rerutning. ")
        return

    wsclean_init = get_options_from_strategy(
        strategy=strategy, mode="wsclean", round="initial"
    )

    if field_options.potato_container:
        preprocess_science_mss = task_potato_peel.map(
            ms=preprocess_science_mss,
            potato_container=field_options.potato_container,
            update_wsclean_options=unmapped(wsclean_init),
        )

    stokes_v_mss = preprocess_science_mss
    wsclean_cmds = task_wsclean_imager.map(
        in_ms=preprocess_science_mss,
        wsclean_container=field_options.wsclean_container,
        update_wsclean_options=unmapped(wsclean_init),
    )
    # TODO: This should be waited!
    beam_summaries = task_create_beam_summary.map(
        ms=preprocess_science_mss, imageset=wsclean_cmds
    )

    archive_wait_for.extend(wsclean_cmds)

    beam_aegean_outputs = None
    if run_aegean:
        beam_aegean_outputs = task_run_bane_and_aegean.map(
            image=wsclean_cmds,
            aegean_container=unmapped(field_options.aegean_container),
        )
        beam_summaries = task_update_with_options.map(
            input_object=beam_summaries, components=beam_aegean_outputs
        )
        field_summary = task_update_with_options.submit(
            input_object=field_summary, beam_summaries=beam_summaries
        )

    if field_options.yandasoft_container:
        parsets = _create_convol_linmos_images(
            wsclean_cmds=wsclean_cmds,
            field_options=field_options,
            field_summary=field_summary,
            current_round=None,
            additional_linmos_suffix_str="poli",
        )
        archive_wait_for.extend(parsets)
        parset = parsets[-1]

        if run_aegean:
            aegean_field_output = task_run_bane_and_aegean.submit(
                image=parset, aegean_container=unmapped(field_options.aegean_container)
            )
            field_summary = task_update_field_summary.submit(
                field_summary=field_summary,
                aegean_outputs=aegean_field_output,
                linmos_command=parset,
            )
            archive_wait_for.append(field_summary)

            if run_validation and field_options.reference_catalogue_directory:
                _validation_items(
                    field_summary=field_summary,
                    aegean_outputs=aegean_field_output,
                    reference_catalogue_directory=field_options.reference_catalogue_directory,
                )

    if field_options.rounds is None:
        logger.info("No self-calibration will be performed. Returning")
        return

    # Set up the default value should the user activated mask option is not set
    fits_beam_masks = None

    for current_round in range(1, field_options.rounds + 1):
        with tags(f"selfcal-{current_round}"):
            final_round = current_round == field_options.rounds

            gain_cal_options = get_options_from_strategy(
                strategy=strategy, mode="gaincal", round=current_round
            )
            wsclean_options = get_options_from_strategy(
                strategy=strategy, mode="wsclean", round=current_round
            )

            skip_gaincal_current_round = consider_skip_selfcal_on_round(
                current_round=current_round,
                skip_selfcal_on_rounds=field_options.skip_selfcal_on_rounds,
            )

            cal_mss = task_gaincal_applycal_ms.map(
                ms=wsclean_cmds,
                round=current_round,
                update_gain_cal_options=unmapped(gain_cal_options),
                archive_input_ms=field_options.zip_ms,
                skip_selfcal=skip_gaincal_current_round,
                rename_ms=field_options.rename_ms,
                archive_cal_table=True,
                wait_for=[
                    field_summary
                ],  # To make sure field summary is created with unzipped MSs
            )
            stokes_v_mss = cal_mss

            fits_beam_masks = None
            if consider_beam_mask_round(
                current_round=current_round,
                mask_rounds=field_options.use_beam_mask_rounds,
                allow_beam_masks=field_options.use_beam_masks,
            ):
                masking_options = get_options_from_strategy(
                    strategy=strategy, mode="masking", round=current_round
                )
                # The is intended to only run the beam wise aegean if it has not alread
                # been done. Immedidatedly after the first round of shallow cleaning
                # aegean could be run.
                beam_aegean_outputs = (
                    task_run_bane_and_aegean.map(
                        image=wsclean_cmds,
                        aegean_container=unmapped(field_options.aegean_container),
                    )
                    if (current_round >= 2 or not beam_aegean_outputs)
                    else beam_aegean_outputs
                )
                fits_beam_masks = task_create_image_mask_model.map(
                    image=wsclean_cmds,
                    image_products=beam_aegean_outputs,
                    min_snr=3.5,
                    update_masking_options=unmapped(masking_options),
                )

            wsclean_cmds = task_wsclean_imager.map(
                in_ms=cal_mss,
                wsclean_container=field_options.wsclean_container,
                update_wsclean_options=unmapped(wsclean_options),
                fits_mask=fits_beam_masks,
            )
            archive_wait_for.extend(wsclean_cmds)

            # Do source finding on the last round of self-cal'ed images
            if round == field_options.rounds and run_aegean:
                task_run_bane_and_aegean.map(
                    image=wsclean_cmds,
                    aegean_container=unmapped(field_options.aegean_container),
                )

            parsets = None  # Without could be unbound
            if field_options.yandasoft_container:
                parsets = _create_convol_linmos_images(
                    wsclean_cmds=wsclean_cmds,
                    field_options=field_options,
                    field_summary=field_summary,
                    current_round=current_round,
                    additional_linmos_suffix_str="poli",
                )
                archive_wait_for.extend(parsets)

            if final_round and run_aegean and parsets:
                aegean_outputs = task_run_bane_and_aegean.submit(
                    image=parsets[-1],
                    aegean_container=unmapped(field_options.aegean_container),
                )
                field_summary = task_update_field_summary.submit(
                    field_summary=field_summary,
                    aegean_outputs=aegean_outputs,
                    round=current_round,
                )
                if run_validation:
                    val_results = _validation_items(
                        field_summary=field_summary,
                        aegean_outputs=aegean_outputs,
                        reference_catalogue_directory=field_options.reference_catalogue_directory,
                    )
                    archive_wait_for.append(val_results)

    if field_options.stokes_v_imaging:
        with tags("stokes-v"):
            stokes_v_wsclean_options = get_options_from_strategy(
                strategy=strategy, mode="wsclean", operation="stokesv"
            )
            wsclean_cmds = task_wsclean_imager.map(
                in_ms=stokes_v_mss,
                wsclean_container=field_options.wsclean_container,
                update_wsclean_options=unmapped(stokes_v_wsclean_options),
                fits_mask=fits_beam_masks,
                wait_for=wsclean_cmds,  # Ensure that measurement sets are doubled up during imaging
            )
            if field_options.yandasoft_container:
                parsets = _create_convol_linmos_images(
                    wsclean_cmds=wsclean_cmds,
                    field_options=field_options.with_options(linmos_residuals=False),
                    field_summary=field_summary,
                    current_round=(
                        field_options.rounds if field_options.rounds else None
                    ),
                    additional_linmos_suffix_str="polv",
                )
                archive_wait_for.extend(parsets)

    # zip up the final measurement set, which is not included in the above loop
    if field_options.zip_ms:
        task_zip_ms.map(in_item=wsclean_cmds)

    if field_options.sbid_archive_path or field_options.sbid_copy_path and run_aegean:
        update_archive_options = get_options_from_strategy(
            strategy=strategy, mode="archive"
        )
        task_archive_sbid.submit(
            science_folder_path=output_split_science_path,
            archive_path=field_options.sbid_archive_path,
            copy_path=field_options.sbid_copy_path,
            max_round=field_options.rounds if field_options.rounds else None,
            update_archive_options=update_archive_options,
            wait_for=archive_wait_for,
        )


def setup_run_process_science_field(
    cluster_config: Union[str, Path],
    science_path: Path,
    split_path: Path,
    field_options: FieldOptions,
    bandpass_path: Optional[Path] = None,
    skip_bandpass_check: bool = False,
) -> None:
    if not skip_bandpass_check and bandpass_path:
        assert (
            bandpass_path.exists() and bandpass_path.is_dir()
        ), f"{bandpass_path=} needs to exist and be a directory! "

    science_sbid = get_sbid_from_path(path=science_path)

    dask_task_runner = get_dask_runner(cluster=cluster_config)

    process_science_fields.with_options(
        name=f"Flint Continuum Pipeline - {science_sbid}", task_runner=dask_task_runner
    )(
        science_path=science_path,
        bandpass_path=bandpass_path,
        split_path=split_path,
        field_options=field_options,
    )

    # TODO: Put the archive stuff here?


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)

    parser.add_argument(
        "--cli-config", is_config_file=True, help="Path to configuration file"
    )

    parser.add_argument(
        "science_path",
        type=Path,
        help="Path to directories containing the beam-wise science measurementsets that will have solutions copied over and applied.",
    )
    parser.add_argument(
        "--calibrated-bandpass-path",
        type=Path,
        default=None,
        help="Path to directory containing the uncalibrated beam-wise measurement sets that contain the bandpass calibration source. If None then the '--sky-model-directory' should be provided. ",
    )
    parser.add_argument(
        "--imaging-strategy",
        type=Path,
        default=None,
        help="Path to a FLINT yaml file that specifies options to use throughout iamging. ",
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        default=Path("."),
        help="Location to write field-split MSs to. Will attempt to use the parent name of a directory when writing out a new MS. ",
    )
    parser.add_argument(
        "--holofile",
        type=Path,
        default=None,
        help="Path to the holography FITS cube used for primary beam corrections",
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
        "--wsclean-container",
        type=Path,
        default=None,
        help="Path to the wsclean singularity container",
    )
    parser.add_argument(
        "--yandasoft-container",
        type=Path,
        default=None,
        help="Path to the singularity container with yandasoft",
    )
    parser.add_argument(
        "--potato-container",
        type=Path,
        default=None,
        help="Path to the potato peel singularity container",
    )
    parser.add_argument(
        "--cluster-config",
        type=str,
        default="petrichor",
        help="Path to a cluster configuration file, or a known cluster name. ",
    )
    parser.add_argument(
        "--selfcal-rounds",
        type=int,
        default=2,
        help="The number of selfcalibration rounds to perfrom. ",
    )
    parser.add_argument(
        "--skip-selfcal-on-rounds",
        type=int,
        nargs="+",
        default=None,
        help="Do not perform the derive and apply self-calibration solutions on these rounds",
    )
    parser.add_argument(
        "--zip-ms",
        action="store_true",
        help="Zip up measurement sets as imaging and self-calibration is carried out.",
    )
    parser.add_argument(
        "--run-aegean",
        action="store_true",
        help="Run the aegean source finder on images. ",
    )
    parser.add_argument(
        "--aegean-container",
        type=Path,
        default=None,
        help="Path to the singularity container with aegean",
    )
    parser.add_argument(
        "--no-imaging",
        action="store_true",
        help="Do not perform any imaging, only derive bandpass solutions and apply to sources. ",
    )
    parser.add_argument(
        "--reference-catalogue-directory",
        type=Path,
        default=None,
        help="Path to the directory containing the ICFS, NVSS and SUMSS referenece catalogues. These are required for validaiton plots. ",
    )
    parser.add_argument(
        "--linmos-residuals",
        action="store_true",
        help="Co-add the per-beam cleaning residuals into a field image",
    )
    parser.add_argument(
        "--beam-cutoff",
        type=float,
        default=150,
        help="Cutoff in arcseconds that is used to flagged synthesised beams were deriving a common resolution to smooth to when forming the linmos images",
    )
    parser.add_argument(
        "--fixed-beam-shape",
        nargs=3,
        type=float,
        default=None,
        help="Specify the final beamsize of linmos field images in (arcsec, arcsec, deg)",
    )
    parser.add_argument(
        "--pb-cutoff",
        type=float,
        default=0.1,
        help="Primary beam attentuation cutoff to use during linmos",
    )
    parser.add_argument(
        "--use-preflagger",
        action="store_true",
        default=False,
        help="Whether to use (or search for solutions with) the preflagger operations applied to the bandpass gain solutions",
    )
    parser.add_argument(
        "--use-beam-masks",
        default=False,
        action="store_true",
        help="Construct a clean mask from an MFS image for the next round of imaging. May adjust some of the imaging options per found if activated. ",
    )
    beam_mask_options = parser.add_mutually_exclusive_group()
    beam_mask_options.add_argument(
        "--use-beam-mask-rounds",
        default=None,
        type=int,
        nargs="+",
        help="If --use-beam-masks is provided, this option specifies from which round of self-calibration the masking operation will be used onwards from. Specific rounds can be set here. ",
    )
    beam_mask_options.add_argument(
        "--use-beam-masks-from",
        default=1,
        type=int,
        help="If --use-beam-masks is provided, this option specifies from which round of self-calibration the masking operation will be used onwards from. ",
    )
    parser.add_argument(
        "--sbid-archive-path",
        type=Path,
        default=None,
        help="Path that SBID archive tarballs will be created under. If None no archive tarballs are created. See ArchiveOptions. ",
    )
    parser.add_argument(
        "--sbid-copy-path",
        type=Path,
        default=None,
        help="Path that final processed products will be copied into. If None no copying of file products is performed. See ArchiveOptions. ",
    )
    parser.add_argument(
        "--skip-bandpass-check",
        default=False,
        action="store_true",
        help="Skip checking whether the path containing bandpass solutions exists (e.g. if solutions have already been applied)",
    )
    parser.add_argument(
        "--rename-ms",
        action="store_true",
        default=False,
        help="Rename MSs throught rounds of imaging and self-cal instead of creating copies. This will delete data-columns throughout. ",
    )
    parser.add_argument(
        "--stokes-v-imaging",
        help="Enables stokes-v imaging after the final round of imaging (whether",
        action="store_true",
        default=False,
    )

    return parser


def cli() -> None:
    import logging

    # logger = logging.getLogger("flint")
    logger.setLevel(logging.INFO)

    parser = get_parser()

    args = parser.parse_args()

    field_options = FieldOptions(
        flagger_container=args.flagger_container,
        calibrate_container=args.calibrate_container,
        holofile=args.holofile,
        expected_ms=args.expected_ms,
        wsclean_container=args.wsclean_container,
        yandasoft_container=args.yandasoft_container,
        potato_container=args.potato_container,
        rounds=args.selfcal_rounds,
        skip_selfcal_on_rounds=args.skip_selfcal_on_rounds,
        zip_ms=args.zip_ms,
        run_aegean=args.run_aegean,
        aegean_container=args.aegean_container,
        no_imaging=args.no_imaging,
        reference_catalogue_directory=args.reference_catalogue_directory,
        linmos_residuals=args.linmos_residuals,
        beam_cutoff=args.beam_cutoff,
        fixed_beam_shape=args.fixed_beam_shape,
        pb_cutoff=args.pb_cutoff,
        use_preflagger=args.use_preflagger,
        use_beam_masks=args.use_beam_masks,
        use_beam_mask_rounds=(
            args.use_beam_mask_rounds
            if args.use_beam_mask_rounds
            else args.use_beam_masks_from
        ),  # defaults value of args.use_beam_masks_from is 1
        imaging_strategy=args.imaging_strategy,
        sbid_archive_path=args.sbid_archive_path,
        sbid_copy_path=args.sbid_copy_path,
        rename_ms=args.rename_ms,
        stokes_v_imaging=args.stokes_v_imaging,
    )

    setup_run_process_science_field(
        cluster_config=args.cluster_config,
        science_path=args.science_path,
        bandpass_path=args.calibrated_bandpass_path,
        split_path=args.split_path,
        field_options=field_options,
        skip_bandpass_check=args.skip_bandpass_check,
    )


if __name__ == "__main__":
    cli()
