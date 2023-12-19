"""A prefect based pipeline that:
- will perform bandpass calibration with PKS B1934-638 data, or from a derived sky-model
- copy and apply to science field
- image and self-calibration the science fields
- run aegean source finding

This pipeline will attempt to incorporate a masking operation once a linmos
field has been produced. Given a linmos image, a signal and mask operation
will create a large field image that could then be used as a clean mask
provided to wsclean. This process would require a mask to be extracted from
the larger singnal linmos mask image using a template WCS header. It seems
that the best way to do this would be to use a header from the preivous
imaging round.
"""
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Optional, Union

from prefect import flow, unmapped

from flint.calibrate.aocalibrate import find_existing_solutions
from flint.logging import logger
from flint.ms import MS
from flint.naming import get_sbid_from_path
from flint.prefect.clusters import get_dask_runner
from flint.prefect.common.imaging import (
    task_convolve_image,
    task_create_apply_solutions_cmd,
    task_create_linmos_mask_model,
    task_create_linmos_mask_wbutter_model,
    task_create_validation_plot,
    task_extract_beam_mask_image,
    task_flag_ms_aoflagger,
    task_gaincal_applycal_ms,
    task_get_common_beam,
    task_linmos_images,
    task_preprocess_askap_ms,
    task_run_bane_and_aegean,
    task_select_solution_for_ms,
    task_split_by_field,
    task_wsclean_imager,
    task_zip_ms,
)
from flint.prefect.common.utils import task_flatten


def _create_linmos_mask(
    linmos_parset: Any, aegean_outputs: Any, butterworth_filter: bool
) -> Any:
    """A common function that pulls together the two linmos masking operations.
    This is an internal function that is calling prefect tasks. It itself can not be
    @task, and is intended not to be share among other codes.
    """
    # At the moment no others tasks or flows should use this, ya filthy pirate.
    if butterworth_filter:
        linmos_mask = task_create_linmos_mask_wbutter_model(
            linmos_parset=linmos_parset,
            image_products=aegean_outputs,
        )
    else:
        linmos_mask = task_create_linmos_mask_model(
            linmos_parset=linmos_parset,
            image_products=aegean_outputs,
        )

    return linmos_mask


@flow(name="Flint Continuum Pipeline")
def process_science_fields(
    science_path: Path,
    bandpass_path: Path,
    split_path: Path,
    calibrate_container: Path,
    flagger_container: Path,
    holofile: Optional[Path] = None,
    expected_ms: int = 36,
    wsclean_container: Optional[Path] = None,
    rounds: Optional[int] = 2,
    yandasoft_container: Optional[Path] = None,
    zip_ms: bool = False,
    run_aegean: bool = False,
    aegean_container: Optional[Path] = None,
    no_imaging: bool = False,
    reference_catalogue_directory: Optional[Path] = None,
    butterworth_filter: bool = True,
) -> None:
    run_aegean = False if aegean_container is None else run_aegean
    run_validation = reference_catalogue_directory is not None

    assert (
        science_path.exists() and science_path.is_dir()
    ), f"{str(science_path)} does not exist or is not a folder. "
    science_mss = list(
        [MS.cast(ms_path) for ms_path in sorted(science_path.glob("*.ms"))]
    )
    assert (
        len(science_mss) == expected_ms
    ), f"Expected to find {expected_ms} in {str(science_path)}, found {len(science_mss)}."

    science_folder_name = science_path.name

    output_split_science_path = (
        Path(split_path / science_folder_name).absolute().resolve()
    )

    if not output_split_science_path.exists():
        logger.info(f"Creating {str(output_split_science_path)}")
        output_split_science_path.mkdir(parents=True)

    logger.info(f"Found the following raw measurement sets: {science_mss}")

    # TODO: This will likely need to be expanded should any
    # other calibration strategies get added
    # Scan the existing bandpass directory for the existing solutions
    calibrate_cmds = find_existing_solutions(
        bandpass_directory=bandpass_path, use_preflagged=True, use_smoothed=False
    )

    logger.info(f"Constructed the following {calibrate_cmds=}")

    split_science_mss = task_split_by_field.map(
        ms=science_mss, field=None, out_dir=unmapped(output_split_science_path)
    )
    # The following line will block until the science
    # fields are split out. Since there might be more
    # than a single field in an SBID, we should do this
    flat_science_mss = task_flatten.submit(split_science_mss)

    preprocess_science_mss = task_preprocess_askap_ms.map(
        ms=flat_science_mss,
        data_column=unmapped("DATA"),
        instrument_column=unmapped("INSTRUMENT_DATA"),
        overwrite=True,
    )
    flag_field_mss = task_flag_ms_aoflagger.map(
        ms=preprocess_science_mss, container=flagger_container, rounds=1
    )
    solutions_paths = task_select_solution_for_ms.map(
        calibrate_cmds=unmapped(calibrate_cmds), ms=flag_field_mss
    )
    apply_solutions_cmds = task_create_apply_solutions_cmd.map(
        ms=flag_field_mss,
        solutions_file=solutions_paths,
        container=calibrate_container,
    )

    if no_imaging:
        logger.info(f"No imaging will be performed, as requested bu {no_imaging=}")
        return

    if wsclean_container is None:
        logger.info("No wsclean container provided. Rerutning. ")
        return

    wsclean_init = {
        "size": 8000,
        "minuv_l": 235,
        "weight": "briggs -0.5",
        "auto_mask": 5,
        "multiscale": True,
        "local_rms_window": 55,
        "multiscale_scales": (0, 15, 30, 40, 50, 60, 70, 120, 240, 480),
    }

    wsclean_cmds = task_wsclean_imager.map(
        in_ms=apply_solutions_cmds,
        wsclean_container=wsclean_container,
        update_wsclean_options=unmapped(wsclean_init),
    )

    task_run_bane_and_aegean.map(
        image=wsclean_cmds, aegean_container=unmapped(aegean_container)
    )

    beam_shape = task_get_common_beam.submit(wsclean_cmds=wsclean_cmds, cutoff=150.0)
    conv_images = task_convolve_image.map(
        wsclean_cmd=wsclean_cmds, beam_shape=unmapped(beam_shape), cutoff=150.0
    )
    parset = task_linmos_images.submit(
        images=conv_images,
        container=yandasoft_container,
        suffix_str="noselfcal",
        holofile=holofile,
    )

    aegean_outputs = task_run_bane_and_aegean.submit(
        image=parset, aegean_container=unmapped(aegean_container)
    )

    linmos_mask = _create_linmos_mask(
        linmos_parset=parset,
        aegean_outputs=aegean_outputs,
        butterworth_filter=butterworth_filter,
    )

    beam_masks = task_extract_beam_mask_image.map(
        linmos_mask_names=unmapped(linmos_mask), wsclean_cmd=wsclean_cmds
    )

    if rounds is None:
        logger.info("No self-calibration will be performed. Returning")
        return

    gain_cal_rounds = {
        1: {"solint": "60s", "uvrange": ">235lambda", "nspw": 1},
        2: {"solint": "20s", "uvrange": ">235lambda", "nspw": 1},
    }
    wsclean_rounds = {
        1: {
            "size": 8000,
            "multiscale": True,
            "minuv_l": 235,
            "auto_mask": 2,
            "local_rms_window": 55,
            "multiscale_scales": (0, 15, 30, 40, 50, 60, 120, 240, 480),
        },
        2: {
            "size": 8000,
            "multiscale": True,
            "minuv_l": 235,
            "auto_mask": 2.0,
            "local_rms_window": 55,
            "multiscale_scales": (0, 15, 30, 40, 50, 60, 120, 240, 480),
        },
    }

    for round in range(1, rounds + 1):
        gain_cal_options = gain_cal_rounds.get(round, None)
        wsclean_options = wsclean_rounds.get(round, None)

        cal_mss = task_gaincal_applycal_ms.map(
            wsclean_cmd=wsclean_cmds,
            round=round,
            update_gain_cal_options=unmapped(gain_cal_options),
            archive_input_ms=zip_ms,
        )

        flag_mss = task_flag_ms_aoflagger.map(
            ms=cal_mss, container=flagger_container, rounds=1
        )
        wsclean_cmds = task_wsclean_imager.map(
            in_ms=flag_mss,
            wsclean_container=wsclean_container,
            update_wsclean_options=unmapped(wsclean_options),
            fits_mask=beam_masks,
        )

        # Do source finding on the last round of self-cal'ed images
        if round == rounds and run_aegean:
            task_run_bane_and_aegean.map(
                image=wsclean_cmds, aegean_container=unmapped(aegean_container)
            )

        beam_shape = task_get_common_beam.submit(
            wsclean_cmds=wsclean_cmds, cutoff=150.0
        )
        conv_images = task_convolve_image.map(
            wsclean_cmd=wsclean_cmds, beam_shape=unmapped(beam_shape), cutoff=150.0
        )
        if yandasoft_container is None:
            logger.info("No yandasoft container supplied, not linmosing. ")
            continue

        parset = task_linmos_images.submit(
            images=conv_images,
            container=yandasoft_container,
            suffix_str=f"round{round}",
            holofile=holofile,
        )

        aegean_outputs = task_run_bane_and_aegean.submit(
            image=parset, aegean_container=unmapped(aegean_container)
        )

        # Use the mask from the first round
        if round < rounds:
            linmos_mask = _create_linmos_mask(
                linmos_parset=parset,
                aegean_outputs=aegean_outputs,
                butterworth_filter=butterworth_filter,
            )

            beam_masks = task_extract_beam_mask_image.map(
                linmos_mask_names=unmapped(linmos_mask), wsclean_cmd=wsclean_cmds
            )

        if run_validation:
            task_create_validation_plot.submit(
                aegean_outputs=aegean_outputs,
                reference_catalogue_directory=reference_catalogue_directory,
            )

        # zip up the final measurement set, which is not included in the above loop
        if round == rounds and zip_ms:
            task_zip_ms.map(in_item=wsclean_cmds)


def setup_run_process_science_field(
    cluster_config: Union[str, Path],
    science_path: Path,
    bandpass_path: Path,
    split_path: Path,
    flagger_container: Path,
    calibrate_container: Path,
    holofile: Optional[Path] = None,
    expected_ms: int = 36,
    wsclean_container: Optional[Path] = None,
    yandasoft_container: Optional[Path] = None,
    rounds: int = 2,
    zip_ms: bool = False,
    run_aegean: bool = False,
    aegean_container: Optional[Path] = None,
    no_imaging: bool = False,
    reference_catalogue_directory: Optional[Path] = None,
    butterworth_filter: bool = False,
) -> None:
    assert (
        bandpass_path.exists() and bandpass_path.is_dir()
    ), f"{bandpass_path=} needs to exist and be a directory! "

    science_sbid = get_sbid_from_path(path=science_path)

    dask_task_runner = get_dask_runner(cluster=cluster_config)

    process_science_fields.with_options(
        name=f"Flint Continuum Masked Pipeline - {science_sbid}",
        task_runner=dask_task_runner,
    )(
        science_path=science_path,
        bandpass_path=bandpass_path,
        split_path=split_path,
        flagger_container=flagger_container,
        calibrate_container=calibrate_container,
        holofile=holofile,
        expected_ms=expected_ms,
        wsclean_container=wsclean_container,
        yandasoft_container=yandasoft_container,
        rounds=rounds,
        zip_ms=zip_ms,
        run_aegean=run_aegean,
        aegean_container=aegean_container,
        no_imaging=no_imaging,
        reference_catalogue_directory=reference_catalogue_directory,
        butterworth_filter=butterworth_filter,
    )


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)

    parser.add_argument(
        "science_path",
        type=Path,
        help="Path to directories containing the beam-wise science measurementsets that will have solutions copied over and applied.",
    )
    parser.add_argument(
        "calibrated_bandpass_path",
        type=Path,
        default=None,
        help="Path to directory containing the calibrated beam-wise measurement sets that contain the bandpass calibration source and their solutions. ",
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
        "--butterworth-filter",
        action="store_true",
        help="Apply a butterworth filter when creating a clean mask from the larger linmos image. This may help capture more extended diffuse emission. ",
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
        science_path=args.science_path,
        bandpass_path=args.calibrated_bandpass_path,
        split_path=args.split_path,
        flagger_container=args.flagger_container,
        calibrate_container=args.calibrate_container,
        holofile=args.holofile,
        expected_ms=args.expected_ms,
        wsclean_container=args.wsclean_container,
        yandasoft_container=args.yandasoft_container,
        rounds=args.selfcal_rounds,
        zip_ms=args.zip_ms,
        run_aegean=args.run_aegean,
        aegean_container=args.aegean_container,
        no_imaging=args.no_imaging,
        reference_catalogue_directory=args.reference_catalogue_directory,
        butterworth_filter=args.butterworth_filter,
    )


if __name__ == "__main__":
    cli()
