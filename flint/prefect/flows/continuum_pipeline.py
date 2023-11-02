"""A prefect based pipeline that:
- will perform bandpass calibration with PKS B1934-638 data, or from a derived sky-model
- copy and apply to science field
- image and self-calibration the science fields
- run aegean source finding
"""
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, List, Dict, Optional, Union, Collection

from prefect import flow, task, unmapped

from flint.bandpass import extract_correct_bandpass_pointing, plot_solutions
from flint.calibrate.aocalibrate import (
    ApplySolutions,
    CalibrateCommand,
    create_apply_solutions_cmd,
    create_calibrate_cmd,
    select_aosolution_for_ms,
    flag_aosolutions,
    find_existing_solutions,
)
from flint.flagging import flag_ms_aoflagger
from flint.imager.wsclean import WSCleanCMD, wsclean_imager
from flint.logging import logger
from flint.ms import MS, preprocess_askap_ms, split_by_field
from flint.prefect.clusters import get_dask_runner
from flint.sky_model import get_1934_model, create_sky_model
from flint.selfcal.casa import gaincal_applycal_ms
from flint.convol import get_common_beam, convolve_images, BeamShape
from flint.coadd.linmos import linmos_images, LinmosCMD
from flint.utils import zip_folder
from flint.source_finding.aegean import run_bane_and_aegean, AegeanOutputs

task_extract_correct_bandpass_pointing = task(extract_correct_bandpass_pointing)
task_preprocess_askap_ms = task(preprocess_askap_ms)
task_flag_ms_aoflagger = task(flag_ms_aoflagger)
task_create_calibrate_cmd = task(create_calibrate_cmd)
task_split_by_field = task(split_by_field)
task_select_solution_for_ms = task(select_aosolution_for_ms)
task_create_apply_solutions_cmd = task(create_apply_solutions_cmd)

@task
def task_bandpass_create_apply_solutions_cmd(
            ms: MS,
            calibrate_cmd: CalibrateCommand,
            container: Path
        ):
    return create_apply_solutions_cmd(ms=ms, solutions_file=calibrate_cmd.solution_path, container=container)
    

@task
def task_run_bane_and_aegean(image: Union[WSCleanCMD,LinmosCMD], aegean_container: Path) -> AegeanOutputs:

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
        logger.info(f"Will be running aegean on a linmos image")

        image_path = image.image_fits
        assert image_path.exists(), f"Image path {image_path} does not exist"
    else:
        raise ValueError(f"Unexpected type, have received {type(image)}. ")

    aegean_outputs = run_bane_and_aegean(image=image_path, aegean_container=aegean_container)

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
    plot_solutions(solutions_path=calibrate_cmd.solution_path, ref_ant=0)


@task
def task_wsclean_imager(
    in_ms: Union[ApplySolutions, MS],
    wsclean_container: Path,
    update_wsclean_options: Optional[Dict[str, Any]] = None,
) -> WSCleanCMD:
    ms = in_ms if isinstance(in_ms, MS) else in_ms.ms

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
    wsclean_cmd: WSCleanCMD, beam_shape: BeamShape, cutoff: float = 25
) -> Collection[Path]:
    assert (
        wsclean_cmd.imageset is not None
    ), f"{wsclean_cmd.ms} has no attached imageset."
    image_paths: Collection[Path] = wsclean_cmd.imageset.image

    logger.info(f"Will convolve {image_paths}")

    return convolve_images(
        image_paths=image_paths, beam_shape=beam_shape, cutoff=cutoff
    )


@task
def task_linmos_images(
    images: Collection[Collection[Path]],
    parset_output_name: str,
    container: Path,
    filter: str = "-MFS-",
    field_name: str = "unnamed_field",
    holofile: Optional[Path] = None,
) -> Path:
    # TODO: Need to flatten images
    # TODO: Need a better way of getting field names

    all_images = [img for beam_images in images for img in beam_images]
    logger.info(f"Number of images to examine {len(all_images)}")

    filter_images = [img for img in all_images if filter in str(img)]
    logger.info(f"Number of filtered images to linmos: {len(filter_images)}")

    out_dir = Path(filter_images[0].parent)
    out_name = out_dir / field_name
    logger.info(f"Base output image name will be: {out_name}")

    linmos_cmd = linmos_images(
        images=filter_images,
        parset_output_name=out_dir / Path(parset_output_name),
        image_output_name=str(out_name),
        container=container,
        holofile=holofile,
    )

    return linmos_cmd.parset


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


def run_bandpass_stage(
    bandpass_mss: Collection[MS],
    output_split_bandpass_path: Path,
    calibrate_container: Path,
    flagger_container: Path,
    model_path: Path,
    source_name_prefix: str = "B1934-638",
    skip_rotation: bool = False
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
        preprocess_bandpass_ms = task_preprocess_askap_ms.submit(ms=extract_bandpass_ms, skip_rotation=skip_rotation)
        flag_bandpass_ms = task_flag_ms_aoflagger.submit(
            ms=preprocess_bandpass_ms, container=flagger_container, rounds=1
        )
        calibrate_cmd = task_create_calibrate_cmd.submit(
            ms=flag_bandpass_ms,
            calibrate_model=model_path,
            container=calibrate_container,
        )
        calibrate_cmd = task_flag_solutions.submit(calibrate_cmd=calibrate_cmd)
        task_plot_solutions.submit(calibrate_cmd=calibrate_cmd)
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
            model_path=model_path,
            use_preflagged=True,
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
    no_imaging: bool = False
) -> None:
    run_aegean = False if aegean_container is None else run_aegean
    
    assert (
        science_path.exists() and science_path.is_dir()
    ), f"{str(science_path)} does not exist or is not a folder. "
    science_mss = list([MS.cast(ms_path) for ms_path in science_path.glob("*.ms")])
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
        "minuv_l": 200,
        "weight": "briggs -0.5",
        "auto_mask": 4.5,
        "multiscale": True,
        "multiscale_scales": (0, 15, 50, 100, 250),
    }
    wsclean_cmds = task_wsclean_imager.map(
        in_ms=apply_solutions_cmd_list,
        wsclean_container=wsclean_container,
        update_wsclean_options=unmapped(wsclean_init),
    )
    if run_aegean:
        task_run_bane_and_aegean.map(image=wsclean_cmds, aegean_container=unmapped(aegean_container))

    beam_shape = task_get_common_beam.submit(wsclean_cmds=wsclean_cmds, cutoff=25.0)
    conv_images = task_convolve_image.map(
        wsclean_cmd=wsclean_cmds, beam_shape=unmapped(beam_shape), cutoff=25.0
    )
    if yandasoft_container is not None:
        parset = task_linmos_images.submit(
            images=conv_images,
            parset_output_name="linmos_noselfcal_parset.txt",
            container=yandasoft_container,
            field_name="example_field_noselfcal",
            holofile=holofile,
        )
        
        if run_aegean:
            task_run_bane_and_aegean.map(image=parset, aegean_container=unmapped(aegean_container))


    if rounds is None:
        logger.info("No self-calibration will be performed. Returning")
        return

    gain_cal_rounds = {
        1: {"solint": "1200s", "uvrange": ">200lambda", "nspw": 1},
        2: {"solint": "60s", "uvrange": ">200lambda", "nspw": 1},
        3: {"solint": "60s", "uvrange": ">200lambda", "nspw": 1},
        4: {"calmode": "ap", "solint": "360s", "uvrange": ">200lambda"},
    }
    wsclean_rounds = {
        1: {"multiscale": True, "minuv_l": 200, "auto_mask": 4, "multiscale_scales": (0, 15, 30, 50, 100, 150)},
        2: {"multiscale": True, "minuv_l": 200, "auto_mask": 3.5, "local_rms_window": 105, "multiscale_scales": (0, 15, 30, 50, 100, 150)},
        3: {"multiscale": False, "minuv_l": 200, "auto_mask": 3.5},
        4: {"multiscale": False, "local_rms_window": 125, "minuv_l": 200, "auto_mask": 3.5},
    }

    for round in range(1, rounds + 1):
        gain_cal_options = gain_cal_rounds.get(round, None)
        wsclean_options = wsclean_rounds.get(round, None)

        cal_mss = task_gaincal_applycal_ms.map(
            wsclean_cmd=wsclean_cmds,
            round=round,
            update_gain_cal_options=unmapped(gain_cal_options),
            archive_input_ms=False,
        )
        if zip_ms:
            task_zip_ms.map(in_item=wsclean_cmds, wait_for=cal_mss)

        flag_mss = task_flag_ms_aoflagger.map(
            ms=cal_mss, container=flagger_container, rounds=1
        )
        wsclean_cmds = task_wsclean_imager.map(
            in_ms=flag_mss,
            wsclean_container=wsclean_container,
            update_wsclean_options=unmapped(wsclean_options),
        )
        
        # Do source finding on the last round of self-cal'ed images
        if round == rounds and run_aegean:
            task_run_bane_and_aegean.map(image=wsclean_cmds, aegean_container=unmapped(aegean_container))

        beam_shape = task_get_common_beam.submit(wsclean_cmds=wsclean_cmds, cutoff=25.0)
        conv_images = task_convolve_image.map(
            wsclean_cmd=wsclean_cmds, beam_shape=unmapped(beam_shape), cutoff=25.0
        )
        if yandasoft_container is None:
            logger.info("No yandasoft container supplied, not linmosing. ")
            continue

        parset = task_linmos_images.submit(
            images=conv_images,
            parset_output_name=f"linmos_round{round}_parset.txt",
            container=yandasoft_container,
            field_name=f"example_field_round{round}",
            holofile=holofile,
        )

        if run_aegean:
            task_run_bane_and_aegean.map(image=parset, aegean_container=unmapped(aegean_container))


    # zip up the final measurement set, which is not included in the above loop
    if zip_ms:
        task_zip_ms.map(in_item=wsclean_cmds, wait_for=cal_mss)


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
    no_imaging: bool = False
) -> None:
    if bandpass_path is None and sky_model_path is None:
        raise ValueError(
            "Both bandpass_path and sky_model_path are None. This is not allowed. "
        )
    if bandpass_path and sky_model_path:
        raise ValueError(
            f"{bandpass_path=} and {sky_model_path=} - one has to be unset. "
        )

    dask_task_runner = get_dask_runner(cluster=cluster_config)

    process_bandpass_science_fields.with_options(
        name="Flint Continuum Pipeline", task_runner=dask_task_runner
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
        no_imaging=no_imaging
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
        help="Do not perform any imaging, only derive bandpass solutions and apply to sources. "
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
        no_imaging=args.no_imaging
    )


if __name__ == "__main__":
    cli()
