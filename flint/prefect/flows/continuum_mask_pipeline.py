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
from typing import Any, Collection, Dict, List, Optional, Union

from prefect import flow, task, unmapped

from flint.bandpass import extract_correct_bandpass_pointing, plot_solutions
from flint.calibrate.aocalibrate import (
    ApplySolutions,
    CalibrateCommand,
    create_apply_solutions_cmd,
    create_calibrate_cmd,
    find_existing_solutions,
    flag_aosolutions,
    select_aosolution_for_ms,
)
from flint.coadd.linmos import LinmosCMD, linmos_images
from flint.convol import BeamShape, convolve_images, get_common_beam
from flint.flagging import flag_ms_aoflagger
from flint.imager.wsclean import WSCleanCMD, wsclean_imager
from flint.logging import logger
from flint.masking import create_snr_mask_from_fits, extract_beam_mask_from_mosaic
from flint.ms import MS, preprocess_askap_ms, split_by_field
from flint.naming import FITSMaskNames, get_sbid_from_path, processed_ms_format
from flint.prefect.clusters import get_dask_runner
from flint.selfcal.casa import gaincal_applycal_ms
from flint.sky_model import create_sky_model, get_1934_model
from flint.source_finding.aegean import AegeanOutputs, run_bane_and_aegean
from flint.utils import zip_folder
from flint.validation import create_validation_plot

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
def task_run_bane_and_aegean(
    image: Union[WSCleanCMD, LinmosCMD], aegean_container: Path
) -> AegeanOutputs:
    if isinstance(image, WSCleanCMD):
        assert image.imageset is not None, "Image set attribute unset. "
        image_paths = image.imageset.image

        logger.info(f"Have extracted image: {image_paths}")

        # For the moment, will only source find on an MFS image
        image_paths = [image for image in image_paths if "-MFS-" in str(image)]
        assert (
            len(image_paths) == 1
        ), "More than one image found after filter for MFS only images. "
        # Get out the only path in the list.
        image_path = image_paths[0]
    elif isinstance(image, LinmosCMD):
        logger.info("Will be running aegean on a linmos image")

        image_path = image.image_fits
        assert image_path.exists(), f"Image path {image_path} does not exist"
    else:
        raise ValueError(f"Unexpected type, have received {type(image)} for {image=}. ")

    aegean_outputs = run_bane_and_aegean(
        image=image_path, aegean_container=aegean_container
    )

    return aegean_outputs


@task
def task_zip_ms(in_item: WSCleanCMD) -> Path:
    ms = in_item.ms

    zipped_ms = zip_folder(in_path=ms.path)

    return zipped_ms


@task
def task_gaincal_applycal_ms(
    wsclean_cmd: WSCleanCMD,
    round: int,
    update_gain_cal_options: Optional[Dict[str, Any]] = None,
    archive_input_ms: bool = False,
) -> MS:
    # TODO: This needs to be expanded to handle multiple MS
    ms = wsclean_cmd.ms

    if not isinstance(ms, MS):
        raise ValueError(
            f"Unsupported {type(ms)=} {ms=}. Likely multiple MS instances? This is not yet supported. "
        )

    return gaincal_applycal_ms(
        ms=ms,
        round=round,
        update_gain_cal_options=update_gain_cal_options,
        archive_input_ms=archive_input_ms,
    )


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


@task
def task_extract_solution_path(calibrate_cmd: CalibrateCommand) -> Path:
    return calibrate_cmd.solution_path


@task
def task_plot_solutions(calibrate_cmd: CalibrateCommand) -> None:
    plot_solutions(solutions_path=calibrate_cmd.solution_path, ref_ant=None)


@task
def task_wsclean_imager(
    in_ms: Union[ApplySolutions, MS],
    wsclean_container: Path,
    update_wsclean_options: Optional[Dict[str, Any]] = None,
    fits_mask: Optional[FITSMaskNames] = None,
) -> WSCleanCMD:
    ms = in_ms if isinstance(in_ms, MS) else in_ms.ms

    update_wsclean_options = (
        {} if update_wsclean_options is None else update_wsclean_options
    )

    if fits_mask:
        update_wsclean_options["fits_mask"] = fits_mask.mask_fits
        update_wsclean_options["auto_mask"] = 2
        update_wsclean_options["force_mask_rounds"] = 2

    logger.info(f"wsclean inager {ms=}")
    return wsclean_imager(
        ms=ms,
        wsclean_container=wsclean_container,
        update_wsclean_options=update_wsclean_options,
    )


@task
def task_get_common_beam(
    wsclean_cmds: Collection[WSCleanCMD], cutoff: float = 25
) -> BeamShape:
    images_to_consider: List[Path] = []

    for wsclean_cmd in wsclean_cmds:
        if wsclean_cmd.imageset is None:
            logger.warn(f"No imageset fo {wsclean_cmd.ms} found. Has imager finished?")
            continue
        images_to_consider.extend(wsclean_cmd.imageset.image)

    logger.info(
        f"Considering {len(images_to_consider)} images across {len(wsclean_cmds)} outputs. "
    )

    beam_shape = get_common_beam(image_paths=images_to_consider, cutoff=cutoff)

    return beam_shape


@task
def task_convolve_image(
    wsclean_cmd: WSCleanCMD, beam_shape: BeamShape, cutoff: float = 60
) -> Collection[Path]:
    assert (
        wsclean_cmd.imageset is not None
    ), f"{wsclean_cmd.ms} has no attached imageset."
    image_paths: Collection[Path] = wsclean_cmd.imageset.image

    logger.info(f"Will convolve {image_paths}")

    # experience has shown that astropy units do not always work correctly
    # in a multiprocessing / dask environment. The unit registery does not
    # seem to serialise correctly, and we can get weird arcsecond is not
    # compatiable with arcsecond type errors. Import here in an attempt
    # to minimise
    import astropy.units as u
    from astropy.io import fits
    from radio_beam import Beam

    # Print the beams out here for logging
    for image_path in image_paths:
        image_beam = Beam.from_fits_header(fits.getheader(str(image_path)))
        logger.info(
            f"{str(image_path.name)}: {image_beam.major.to(u.arcsecond)} {image_beam.minor.to(u.arcsecond)}  {image_beam.pa}"
        )

    return convolve_images(
        image_paths=image_paths, beam_shape=beam_shape, cutoff=cutoff
    )


@task
def task_linmos_images(
    images: Collection[Collection[Path]],
    container: Path,
    filter: str = "-MFS-",
    field_name: Optional[str] = None,
    suffix_str: str = "noselfcal",
    holofile: Optional[Path] = None,
    sbid: Optional[int] = None,
    parset_output_path: Optional[str] = None,
) -> LinmosCMD:
    # TODO: Need to flatten images
    # TODO: Need a better way of getting field names

    all_images = [img for beam_images in images for img in beam_images]
    logger.info(f"Number of images to examine {len(all_images)}")

    filter_images = [img for img in all_images if filter in str(img)]
    logger.info(f"Number of filtered images to linmos: {len(filter_images)}")

    candidate_image = filter_images[0]
    candidate_image_fields = processed_ms_format(in_name=candidate_image)

    if field_name is None:
        field_name = candidate_image_fields.field
        logger.info(f"Extracted {field_name=} from {candidate_image=}")

    if sbid is None:
        sbid = candidate_image_fields.sbid
        logger.info(f"Extracted {sbid=} from {candidate_image=}")

    base_name = f"SB{sbid}.{field_name}.{suffix_str}"

    out_dir = Path(filter_images[0].parent)
    out_name = out_dir / base_name
    logger.info(f"Base output image name will be: {out_name}")

    if parset_output_path is None:
        parset_output_path = f"{out_name.name}_parset.txt"

    parset_output_path = out_dir / Path(parset_output_path)
    logger.info(f"Parsert output path is {parset_output_path}")

    linmos_cmd = linmos_images(
        images=filter_images,
        parset_output_path=Path(parset_output_path),
        image_output_name=str(out_name),
        container=container,
        holofile=holofile,
    )

    return linmos_cmd


@task
def task_create_linmos_mask_model(
    linmos_parset: LinmosCMD, image_products: AegeanOutputs
) -> FITSMaskNames:
    if isinstance(image_products, AegeanOutputs):
        linmos_image = linmos_parset.image_fits
        linmos_rms = image_products.rms
        linmos_bkg = image_products.bkg
    else:
        raise ValueError("Unsupported bkg/rms mode. ")

    logger.info(f"Creating a clean mask for {linmos_image=}")
    logger.info(f"Using {linmos_rms=}")
    logger.info(f"Using {linmos_bkg=}")

    linmos_mask_names = create_snr_mask_from_fits(
        fits_image_path=linmos_image,
        fits_bkg_path=linmos_bkg,
        fits_rms_path=linmos_rms,
        create_signal_fits=True,
        min_snr=3,
    )

    logger.info(f"Created {linmos_mask_names.mask_fits}")

    return linmos_mask_names


@task
def task_extract_beam_mask_image(
    linmos_mask_names: FITSMaskNames, wsclean_cmd: WSCleanCMD
) -> FITSMaskNames:
    # All images made by wsclean will have the same WCS
    beam_image = wsclean_cmd.imageset.image[0]
    beam_mask_names = extract_beam_mask_from_mosaic(
        fits_beam_image_path=beam_image, fits_mosaic_mask_names=linmos_mask_names
    )

    return beam_mask_names


@task
def task_create_sky_model(
    ms: MS, cata_dir: Path, calibrate_container: Path
) -> CalibrateCommand:
    sky_model = create_sky_model(
        ms_path=ms.path, cata_dir=cata_dir, hyperdrive_model=False
    )

    # To ensure the linter in the create calibrate command below is happy
    assert isinstance(
        sky_model.calibrate_model, Path
    ), "Only calibrate models supported, and not set in sky_model. "

    calibrate_command = create_calibrate_cmd(
        ms=ms, calibrate_model=sky_model.calibrate_model, container=calibrate_container
    )

    return calibrate_command


@task
def task_create_validation_plot(
    aegean_outputs: AegeanOutputs, reference_catalogue_directory: Path
) -> Path:
    output_figure_path = aegean_outputs.comp.with_suffix(".validation.png")

    logger.info(f"Will create {output_figure_path=}")

    return create_validation_plot(
        rms_image_path=aegean_outputs.rms,
        source_catalogue_path=aegean_outputs.comp,
        output_path=output_figure_path,
        reference_catalogue_directory=reference_catalogue_directory,
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
    if not output_split_bandpass_path.exists():
        logger.info(f"Creating {str(output_split_bandpass_path)}")
        output_split_bandpass_path.mkdir(parents=True)

    calibrate_cmds: List[CalibrateCommand] = []

    for bandpass_ms in bandpass_mss:
        extract_bandpass_ms = task_extract_correct_bandpass_pointing.submit(
            ms=bandpass_ms,
            source_name_prefix=source_name_prefix,
            ms_out_dir=output_split_bandpass_path,
        )
        preprocess_bandpass_ms = task_preprocess_askap_ms.submit(
            ms=extract_bandpass_ms, skip_rotation=skip_rotation
        )
        flag_bandpass_ms = task_flag_ms_aoflagger.submit(
            ms=preprocess_bandpass_ms, container=flagger_container, rounds=1
        )
        calibrate_cmd = task_create_calibrate_cmd.submit(
            ms=flag_bandpass_ms,
            calibrate_model=model_path,
            container=calibrate_container,
        )
        calibrate_cmd = task_flag_solutions.submit(calibrate_cmd=calibrate_cmd)
        calibrate_cmds.append(calibrate_cmd)

        apply_solutions_cmd = task_bandpass_create_apply_solutions_cmd.submit(
            ms=flag_bandpass_ms,
            calibrate_cmd=calibrate_cmd,
            container=calibrate_container,
        )

    return calibrate_cmds


def run_bandpass_calibration(
    bandpass_path: Path,
    split_path: Path,
    expected_ms: int,
    calibrate_container: Path,
    flagger_container: Path,
    source_name_prefix: str = "B1934-638",
):
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

    # TODO: This check currently expects the input bandpass_path to reffer
    # to the raw MS data. The output_split_bandpass_path is built up from
    # that. This behaviour should (or could?) be dependent on a different
    # option to explictly set the location of precomputed solutions.
    if output_split_bandpass_path.exists():
        logger.info(
            (
                f"The output bandpass folder {output_split_bandpass_path} appears to exist. "
                "Will construct commands from pre-computed solutions. "
            )
        )
        # TODO: This will likely need to be expanded should any
        # other calibration strategies get added
        calibrate_cmds = find_existing_solutions(
            bandpass_directory=output_split_bandpass_path,
            use_preflagged=True,
            use_smoothed=False,
        )
    else:
        logger.info(
            f"Output bandpass directory {output_split_bandpass_path} not found. Will process the bandpass data. "
        )
        calibrate_cmds = run_bandpass_stage(
            bandpass_mss=bandpass_mss,
            output_split_bandpass_path=output_split_bandpass_path,
            calibrate_container=calibrate_container,
            flagger_container=flagger_container,
            model_path=model_path,
            source_name_prefix=source_name_prefix,
        )

    return calibrate_cmds


@flow(name="Flint Continuum Pipeline")
def process_bandpass_science_fields(
    science_path: Path,
    split_path: Path,
    flagger_container: Path,
    calibrate_container: Path,
    holofile: Optional[Path] = None,
    expected_ms: int = 36,
    source_name_prefix: str = "B1934-638",
    wsclean_container: Optional[Path] = None,
    rounds: Optional[int] = 2,
    yandasoft_container: Optional[Path] = None,
    bandpass_path: Optional[Path] = None,
    sky_model_path: Optional[Path] = None,
    zip_ms: bool = False,
    run_aegean: bool = False,
    aegean_container: Optional[Path] = None,
    no_imaging: bool = False,
    reference_catalogue_directory: Optional[Path] = None,
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

    calibrate_cmds = None
    if bandpass_path:
        calibrate_cmds = run_bandpass_calibration(
            bandpass_path=bandpass_path,
            split_path=split_path,
            expected_ms=expected_ms,
            calibrate_container=calibrate_container,
            flagger_container=flagger_container,
            source_name_prefix=source_name_prefix,
        )

    science_fields = task_split_by_field.map(
        ms=science_mss, field=None, out_dir=unmapped(output_split_science_path)
    )

    # The following line will block until the science
    # fields are split out. Since there might be more
    # than a single field in an SBID, we should do this
    field_science_mss = task_flatten_prefect_futures(science_fields)

    apply_solutions_cmd_list = []

    for field_science_ms in field_science_mss:
        logger.info(f"Processing {field_science_ms}.")
        preprocess_science_ms = task_preprocess_askap_ms.submit(
            ms=field_science_ms,
            data_column="DATA",
            instrument_column="INSTRUMENT_DATA",
            overwrite=True,
        )
        flag_field_ms = task_flag_ms_aoflagger.submit(
            ms=preprocess_science_ms, container=flagger_container, rounds=1
        )

        if sky_model_path:
            calibrate_cmd = task_create_sky_model.submit(
                ms=flag_field_ms,
                cata_dir=sky_model_path,
                calibrate_container=calibrate_container,
            )
            calibrate_cmd = task_flag_solutions.submit(calibrate_cmd=calibrate_cmd)
            task_plot_solutions.submit(calibrate_cmd=calibrate_cmd)

            solutions_path = task_extract_solution_path.submit(
                calibrate_cmd=calibrate_cmd
            )

        elif bandpass_path and calibrate_cmds is not None:
            solutions_path = task_select_solution_for_ms.submit(
                calibrate_cmds=calibrate_cmds, ms=flag_field_ms, wait_for=calibrate_cmds
            )
        else:
            raise ValueError(
                "Neither a bandpass calibration or sky-model calibration procedure set. "
            )

        apply_solutions_cmd = task_create_apply_solutions_cmd.submit(
            ms=flag_field_ms,
            solutions_file=solutions_path,
            container=calibrate_container,
        )
        apply_solutions_cmd_list.append(apply_solutions_cmd)

    if no_imaging:
        logger.info(f"No imaging will be performed, as requested bu {no_imaging=}")
        return

    if wsclean_container is None:
        logger.info("No wsclean container provided. Rerutning. ")
        return

    wsclean_init = {
        "size": 6644,
        "minuv_l": 235,
        "weight": "briggs 0.5",
        "auto_mask": 5,
        "multiscale": True,
        "local_rms_window": 55,
        "multiscale_scales": (0, 15, 30, 40, 50, 60, 70, 120, 240),
    }

    wsclean_cmds = task_wsclean_imager.map(
        in_ms=apply_solutions_cmd_list,
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

    linmos_mask = task_create_linmos_mask_model(
        linmos_parset=parset,
        image_products=aegean_outputs,
    )

    beam_masks = task_extract_beam_mask_image.map(
        linmos_mask_names=unmapped(linmos_mask), wsclean_cmd=wsclean_cmds
    )

    if rounds is None:
        logger.info("No self-calibration will be performed. Returning")
        return

    gain_cal_rounds = {
        1: {"solint": "1200s", "uvrange": ">235lambda", "nspw": 1},
        2: {"solint": "60s", "uvrange": ">235lambda", "nspw": 1},
    }
    wsclean_rounds = {
        1: {
            "size": 6644,
            "multiscale": True,
            "minuv_l": 235,
            "auto_mask": 5,
            "local_rms_window": 55,
            "multiscale_scales": (0, 15, 30, 40, 50, 60, 120, 240),
        },
        2: {
            "size": 6644,
            "multiscale": True,
            "minuv_l": 235,
            "auto_mask": 4.0,
            "local_rms_window": 55,
            "multiscale_scales": (0, 15, 30, 40, 50, 60, 120, 240),
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
        # if round < rounds:
        #     linmos_mask = task_create_linmos_mask_model.submit(
        #         linmos_parset=parset,
        #         image_products=aegean_outputs,
        #     )

        #     beam_masks = task_extract_beam_mask_image.map(
        #         linmos_mask_names=unmapped(linmos_mask), wsclean_cmd=wsclean_cmds
        #     )

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
    split_path: Path,
    flagger_container: Path,
    calibrate_container: Path,
    holofile: Optional[Path] = None,
    expected_ms: int = 36,
    source_name_prefix: str = "B1934-638",
    wsclean_container: Optional[Path] = None,
    yandasoft_container: Optional[Path] = None,
    rounds: int = 2,
    bandpass_path: Optional[Path] = None,
    sky_model_path: Optional[Path] = None,
    zip_ms: bool = False,
    run_aegean: bool = False,
    aegean_container: Optional[Path] = None,
    no_imaging: bool = False,
    reference_catalogue_directory: Optional[Path] = None,
) -> None:
    if bandpass_path is None and sky_model_path is None:
        raise ValueError(
            "Both bandpass_path and sky_model_path are None. This is not allowed. "
        )
    if bandpass_path and sky_model_path:
        raise ValueError(
            f"{bandpass_path=} and {sky_model_path=} - one has to be unset. "
        )

    science_sbid = get_sbid_from_path(path=science_path)

    dask_task_runner = get_dask_runner(cluster=cluster_config)

    process_bandpass_science_fields.with_options(
        name=f"Flint Continuum Masked Pipeline - {science_sbid}",
        task_runner=dask_task_runner,
    )(
        science_path=science_path,
        split_path=split_path,
        flagger_container=flagger_container,
        calibrate_container=calibrate_container,
        holofile=holofile,
        expected_ms=expected_ms,
        source_name_prefix=source_name_prefix,
        wsclean_container=wsclean_container,
        yandasoft_container=yandasoft_container,
        rounds=rounds,
        bandpass_path=bandpass_path,
        sky_model_path=sky_model_path,
        zip_ms=zip_ms,
        run_aegean=run_aegean,
        aegean_container=aegean_container,
        no_imaging=no_imaging,
        reference_catalogue_directory=reference_catalogue_directory,
    )


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)

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
        "--sky-model-path",
        type=Path,
        default=None,
        help="Path to the directory containing knows sky-model catalogue files that can be used to derive an estimated in-field sky-model. If None, a --bandpass-path should be provided. ",
    )
    parser.add_argument(
        "--bandpass-path",
        type=Path,
        default=None,
        help="Path to directory containing the uncalibrated beam-wise measurement sets that contain the bandpass calibration source. If None then the '--sky-model-directory' should be provided. ",
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
        split_path=args.split_path,
        flagger_container=args.flagger_container,
        calibrate_container=args.calibrate_container,
        holofile=args.holofile,
        expected_ms=args.expected_ms,
        wsclean_container=args.wsclean_container,
        yandasoft_container=args.yandasoft_container,
        rounds=args.selfcal_rounds,
        bandpass_path=args.bandpass_path,
        sky_model_path=args.sky_model_path,
        zip_ms=args.zip_ms,
        run_aegean=args.run_aegean,
        aegean_container=args.aegean_container,
        no_imaging=args.no_imaging,
        reference_catalogue_directory=args.reference_catalogue_directory,
    )


if __name__ == "__main__":
    cli()