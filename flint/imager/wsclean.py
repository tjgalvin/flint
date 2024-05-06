"""Simple interface into wsclean"""

from __future__ import annotations

from argparse import ArgumentParser
from glob import glob
from numbers import Number
from pathlib import Path
from typing import Any, Collection, Dict, List, NamedTuple, Optional, Tuple, Union

from flint.exceptions import CleanDivergenceError
from flint.logging import logger
from flint.ms import MS
from flint.sclient import run_singularity_command


class ImageSet(NamedTuple):
    """A structure to represent the images and auxillary products produced by
    wsclean"""

    prefix: str
    """Prefix of the images and other output products. This should correspond to the -name argument from wsclean"""
    image: Collection[Path]
    """Images produced. """
    psf: Optional[Collection[Path]] = None
    """References to the PSFs produced by wsclean. """
    dirty: Optional[Collection[Path]] = None
    """Dirty images. """
    model: Optional[Collection[Path]] = None
    """Model images.  """
    residual: Optional[Collection[Path]] = None
    """Residual images."""


class WSCleanOptions(NamedTuple):
    """A basic container to handle WSClean options. These attributes should
    conform to the same option name in the calling signature of wsclean
    """

    abs_mem: int = 100
    """Memory wsclean should try to limit itself to"""
    local_rms_window: int = 65
    """Size of the window used to estimate rms noise"""
    size: int = 10128
    """Image size, only a single dimension is required. Note that this means images will be squares. """
    local_rms: bool = True
    """Whether a local rms map is computed"""
    force_mask_rounds: int = 10
    """Round of force masked derivation"""
    auto_mask: float = 3.5
    """How deep the construct clean mask is during each cycle"""
    auto_threshold: float = 0.5
    """How deep to clean once initial clean threshold reached"""
    threshold: Optional[float] = None
    """Threshold in Jy to stop cleaning"""
    channels_out: int = 4
    """Number of output channels"""
    mgain: float = 0.7
    """Major cycle gain"""
    nmiter: int = 15
    """Maximum number of major cycles to perform"""
    niter: int = 750000
    """Maximum numer of minor cycles"""
    multiscale: bool = True
    """Enable multiscale deconvolution"""
    multiscale_scale_bias: float = 0.75
    """Multiscale bias term"""
    multiscale_scales: Optional[Collection[int]] = (
        0,
        15,
        25,
        50,
        75,
        100,
        250,
        400,
    )
    """Scales used for multi-scale deconvolution"""
    fit_spectral_pol: int = 2
    """Number of spectral terms to include during sub-band subtractin"""
    weight: str = "briggs -0.5"
    """Robustness of the weighting used"""
    data_column: str = "CORRECTED_DATA"
    """Which column in the MS to image"""
    scale: str = "2.5asec"
    """Pixel scale size"""
    gridder: Optional[str] = "wgridder"
    """Use the wgridder kernel in wsclean (instead of the default w-stacking method)"""
    nwlayers: Optional[int] = None
    """Number of w-layers to use if the gridder mode is w-stacking"""
    wgridder_accuracy: float = 1e-4
    """The accuracy requested of the wgridder (should it be used), compared as the RMS error when compred to a DFT"""
    join_channels: bool = True
    """Collapse the sub-band images down to an MFS image when peak-finding"""
    minuv_l: Optional[float] = None
    """The minimum lambda length that the visibility data needs to meet for it to be selected for imaging"""
    minuvw_m: Optional[float] = None
    """A (u,v) selection command, where any baselines shorter than this will be ignored during imaging"""
    maxw: Optional[float] = None
    """A percentage specifying the maximum w-term to be gridded, relative to the max w-term being considered"""
    no_update_model_required: bool = False
    """Will instruct wsclean not to create the MODEL_DATA column"""
    no_small_inversion: bool = False
    """Disables an optimisation of wsclean's w-gridder mode. This might improve accuracy of the w-gridder. """
    name: Optional[str] = None
    """Name of the output files passed through to wsclean"""
    beam_fitting_size: Optional[float] = 1.25
    """Use a fitting box the size of <factor> times the theoretical beam size for fitting a Gaussian to the PSF."""
    fits_mask: Optional[Path] = None
    """Path to a FITS file that encodes a cleaning mask"""
    deconvolution_channels: Optional[int] = None
    """The channels out will be averaged down to this many sub-band images during deconvolution"""
    parallel_deconvolution: Optional[int] = None
    """If not none, then this is the number of sub-regions wsclean will attempt to divide and clean"""
    parallel_gridding: Optional[int] = None
    """If not none, then this is the number of channel images that will be gridded in parallel"""

    def with_options(self, **kwargs) -> WSCleanOptions:
        """Return a new instance of WSCleanOptions with updated components"""
        _dict = self._asdict()
        _dict.update(**kwargs)

        return WSCleanOptions(**_dict)


class WSCleanCommand(NamedTuple):
    """Simple container for a wsclean command."""

    cmd: str
    """The constructede wsclean command that would be executed."""
    options: WSCleanOptions
    """The set of wslean options used for imaging"""
    ms: Union[MS, Tuple[MS]]
    """The measurement sets that have been included in the wsclean command. """
    imageset: Optional[ImageSet] = None
    """Collection of images produced by wsclean"""
    cleanup: bool = True
    """Will clean up the dirty images/psfs/residuals/models when the imaging has completed"""

    def with_options(self, **kwargs) -> WSCleanCommand:
        _dict = self._asdict()
        _dict.update(**kwargs)

        return WSCleanCommand(**_dict)


def _wsclean_output_callback(line: str) -> None:
    """Call back function used to detect clean divergence"""

    assert isinstance(line, str)

    if "Iteration" in line and "KJy" in line:
        raise CleanDivergenceError(f"Clean divergence detected: {line}")


def get_wsclean_output_names(
    prefix: str,
    subbands: int,
    pols: Optional[Union[str, Collection[str]]] = None,
    verify_exists: bool = False,
    include_mfs: bool = True,
    output_types: Union[str, Collection[str]] = (
        "image",
        "dirty",
        "residual",
        "model",
        "psf",
    ),
    check_exists_when_adding: bool = False,
) -> ImageSet:
    """Attempt to generate the file names and paths that would be
    created by an imaging run of wsclean. This is done using a the
    known (expected) naming format of modern wsclean versions.

    Checks can be made to ensure a file exists before adding it
    to the output ``ImageSet``. This might be important as some
    wsclean image products might be deleted in order to preserve
    disk space.

    Args:
        prefix (str): The prefix of the imaging run (the -name option in wsclean call)
        subbands (int): Number of subbands that were imaged
        pol (Optional[Union[str,Collection[str]]], optional): The polarisation of the image. If None are provided then this is not used. Multiple polarisation may be supplied. If multiple pols are given in an iterable, each will be produced. Defaults to None.
        verify_exists (bool, optional): Ensures that each generated path corresponds to an actual file. Defaults to False.
        include_mfs (bool, optional): Include the MFS images produced by wsclean. Defaults to True.
        output_types (Union[str,Collection[str]]): Include files of this type, including image, dirty, residual, model, psf. Defaults to  ('image','dirty','residual','model', 'psf').
        check_exists_when_adding (bool, optional): Only add a generated file if it exists. Although related to the ``verify_exists``, when this optional is ``True`` files will silently be ignored if they are not found. if this and ``verify_exists`` are both ``True`` then no errors on missing files will be raised. Defaults to False.

    Raises:
        FileExistsError: Raised when a file does not exist and verify_exists is True.

    Returns:
        ImageSet: The file paths that wsclean should create/has created.
    """
    # TODO: NEED TESTS!
    subband_strs = [f"{subband:04}" for subband in range(subbands)]
    if include_mfs:
        subband_strs.append("MFS")

    in_pols: Tuple[Union[None, str]]
    if pols is None:
        in_pols = (None,)
    elif isinstance(pols, str):
        in_pols = (pols,)
    else:
        in_pols = pols

    if isinstance(output_types, str):
        output_types = (output_types,)

    images: Dict[str, Collection[Path]] = {}
    for image_type in ("image", "dirty", "model", "residual"):
        if image_type not in output_types:
            continue

        paths: List[Path] = []
        for pol in in_pols:
            for subband_str in subband_strs:
                if pol:
                    path_str = f"{prefix}-{subband_str}-{pol}-{image_type}.fits"
                else:
                    path_str = f"{prefix}-{subband_str}-{image_type}.fits"

                if check_exists_when_adding and not Path(path_str).exists():
                    logger.debug(f"{path_str} does not existing. Not adding. ")
                    continue

                paths.append(Path(path_str))

        images[image_type] = paths

    # The PSF is the same for all stokes
    if "psf" in output_types:
        psf_images = [
            Path(f"{prefix}-{subband_str}-psf.fits") for subband_str in subband_strs
        ]
        # Filter out files if they do not exists
        if check_exists_when_adding:
            psf_images = [psf_image for psf_image in psf_images if psf_image.exists()]

        images["psf"] = psf_images

    if verify_exists:
        paths_no_exists: List[Path] = []
        for _, check_paths in images.items():
            paths_no_exists += [path for path in check_paths if not path.exists()]
        if len(paths_no_exists) > 0:
            raise FileExistsError(
                f"The following {len(paths_no_exists)} files do not exist: {paths_no_exists}"
            )

    return ImageSet(prefix=prefix, **images)


def delete_wsclean_outputs(
    prefix: str, output_type: str = "image", ignore_mfs: bool = True
) -> Collection[Path]:
    """Attempt to remove elected wsclean output files

    Args:
        prefix (str): The prefix of the files to remove. This would correspond to the -name of wsclean.
        output_type (str, optional): What type of wsclean output to try to remove. Defaults to 'image'.
        ignore_mfs (bool, optional): If True, do not remove MFS outputs (attempt to, atleast). Defaults to True.

    Returns:
        Collection[Path]: The paths that were removed (or at least attempted to be removed)/
    """
    # TODO: This glob needs to be replaced with something more explicit
    paths = [Path(p) for p in glob(f"{prefix}-*{output_type}.fits")]
    logger.info(f"Found {len(paths)} matching {prefix=} and {output_type=}.")
    rm_paths: List[Path] = []

    for path in paths:
        if ignore_mfs and "-MFS-" in str(path.name):
            logger.info(f"{path} appears to be an MFS product, not removing. ")
            continue
        if path.exists():
            logger.warning(f"Removing {path}.")
            try:
                rm_paths.append(path)
                path.unlink()
            except Exception as e:
                logger.critical(f"Removing {path} failed: {e}")

    return rm_paths


def create_wsclean_cmd(
    ms: MS, wsclean_options: WSCleanOptions, container: Optional[Path] = None
) -> WSCleanCommand:
    """Create a wsclean command from a WSCleanOptions container

    Args:
        ms (MS): The measurement set to be imaged
        wsclean_options (WSCleanOptions): WSClean options to image with
        container (Optional[Path], optional): If a path to a container is provided the command is executed immediatedly. Defaults to None.

    Raises:
        ValueError: Raised when a option has not been successfully processed

    Returns:
        WSCleanCommand: The wsclean command to run
    """
    # Some wsclean options, if multiple values are provided, might need
    # to be join as a csv list. Others might want to be dumped in. Just
    # attempting to future proof (arguably needlessly).
    options_to_comma_join = "multiscale-scales"

    cmd = "wsclean "
    unknowns: List[Tuple[Any, Any]] = []
    logger.info("Creating wsclean command.")
    for key, value in wsclean_options._asdict().items():
        key = key.replace("_", "-")
        logger.debug(f"{key=} {value=} {type(value)=}")
        if key == "size":
            cmd += f"-size {value} {value} "
        elif key == "wgridder-accuracy":
            if wsclean_options.gridder == "wgridder":
                cmd += f"-{key} {value} "
        elif isinstance(value, bool):
            if value:
                cmd += f"-{key} "
        elif isinstance(value, (str, Number)):
            cmd += f"-{key} {value} "
        elif isinstance(value, (list, tuple)):
            value = list(map(str, value))
            value_str = (
                ",".join(value) if key in options_to_comma_join else " ".join(value)
            )
            cmd += f"-{key} {value_str} "
        elif isinstance(value, Path):
            value_str = str(value)
            cmd += f"-{key} {value_str} "
        elif value is None:
            logger.debug(
                f"{key} option set to {value}. Not sure what this means. Ignoring. "
            )
        else:
            unknowns.append((key, value))

    if len(unknowns) > 0:
        msg = ", ".join([f"{t[0]} {t[1]}" for t in unknowns])
        raise ValueError(f"Unknown wsclean option types: {msg}")

    cmd += f"{str(ms.path)} "

    logger.info(f"Constructed wsclean command: {cmd=}")
    logger.info("Setting default model data column to 'MODEL_DATA'")
    wsclean_cmd = WSCleanCommand(
        cmd=cmd, options=wsclean_options, ms=ms.with_options(model_column="MODEL_DATA")
    )

    if container:
        wsclean_cmd = run_wsclean_imager(wsclean_cmd=wsclean_cmd, container=container)

    return wsclean_cmd


def run_wsclean_imager(wsclean_cmd: WSCleanCommand, container: Path) -> WSCleanCommand:
    """Run a provided wsclean command. Optionally will clean up files,
    including the dirty beams, psfs and other assorted things.

    Args:
        wsclean_cmd (WSCleanCommand): The command to run, and other properties (cleanup.)
        container (Path): Path to the container with wsclean available in it

    Returns:
        WSCleanCommand: The executed wsclean command with a populated imageset properter.
    """

    ms = wsclean_cmd.ms
    if isinstance(ms, MS):
        ms = (ms,)

    bind_dirs = [Path(m.path).parent.absolute() for m in ms]
    run_singularity_command(
        image=container,
        command=wsclean_cmd.cmd,
        bind_dirs=bind_dirs,
        stream_callback_func=_wsclean_output_callback,
    )

    prefix = wsclean_cmd.options.name
    if prefix is None:
        prefix = ms[0].path.name
        logger.warning(f"Setting prefix to {prefix}. Likely this is not correct. ")

    if wsclean_cmd.cleanup:
        logger.info("Will clean up files created by wsclean. ")

        for output_type in ("dirty", "psf", "model", "residual"):
            delete_wsclean_outputs(prefix=prefix, output_type=output_type)
        # for output_type in ("model", "residual"):
        #     delete_wsclean_outputs(
        #         prefix=prefix, output_type=output_type, ignore_mfs=False
        #     )

    imageset = get_wsclean_output_names(
        prefix=prefix,
        subbands=wsclean_cmd.options.channels_out,
        verify_exists=True,
        output_types=("image", "residual"),
        check_exists_when_adding=True,
    )

    logger.info(f"Constructed {imageset=}")

    return wsclean_cmd.with_options(imageset=imageset)


def wsclean_imager(
    ms: Union[Path, MS],
    wsclean_container: Path,
    wsclean_options_path: Optional[Path] = None,
    update_wsclean_options: Optional[Dict[str, Any]] = None,
) -> WSCleanCommand:
    """Create and run a wsclean imager command against a measurement set.

    Args:
        ms (Union[Path,MS]): Path to the measurement set that will be imaged
        wsclean_container (Path): Path to the container with wsclean installed
        wsclean_options_path (Optional[Path], optional): Location of a wsclean set of options. Defaults to None.
        update_wsclean_options (Optional[Dict[str, Any]], optional): Additional options to update the generated WscleanOptions with. Keys should be attributes of WscleanOptions. Defaults ot None.

    Returns:
        WSCleanCommand: _description_
    """

    # TODO: This should be expanded to support multiple measurement sets
    ms = MS.cast(ms)

    if wsclean_options_path:
        logger.warning(
            "This is a place holder for loading a wsclean imager parameter file. It is being ignored. "
        )

    wsclean_options = WSCleanOptions()
    if update_wsclean_options is not None:
        logger.info("Updatting wsclean options with user-provided items. ")
        wsclean_options = wsclean_options.with_options(**update_wsclean_options)

    if wsclean_options.name is None:
        # TODO: Come up with a consistent naming scheme. Add in a naming submodule
        # to consolidate this functionality
        wsclean_name = ms.path.absolute().parent / ms.path.stem
        logger.warning(f"Autogenerated wsclean output name: {wsclean_name}")
        wsclean_options = wsclean_options.with_options(name=str(wsclean_name))

    assert ms.column is not None, "A MS column needs to be elected for imaging. "
    wsclean_options = wsclean_options.with_options(data_column=ms.column)
    wsclean_cmd = create_wsclean_cmd(
        ms=ms, wsclean_options=wsclean_options, container=wsclean_container
    )

    return wsclean_cmd


def create_template_wsclean_options(
    input_wsclean_options: WSCleanOptions,
) -> WSCleanOptions:
    """Construct a simple instance of WSClean options that will not
    actually clean. This is intended to be used to get a representations
    FITS header with appropriate WSC information.

    Args:
        input_wsclean_options (WSCleanOptions): The base set of wsclean options to use

    Returns:
        WSCleanOptions: Template options to use for the wsclean fits header creation
    """

    template_options = WSCleanCommand(
        size=input_wsclean_options.size,
        channels_out=1,
        nmiter=0,
        niter=1,
        data_column=input_wsclean_options.data_column,
        scale=input_wsclean_options.scale,
        name=f"{input_wsclean_options.name}_template",
    )
    logger.info(f"Template options are {template_options}")

    return template_options


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Routines related to wsclean")

    subparser = parser.add_subparsers(dest="mode")

    wsclean_parser = subparser.add_parser(
        "image", help="Attempt to run a wsclean commmand. "
    )
    wsclean_parser.add_argument(
        "ms", type=Path, help="Path to a measurement set to image"
    )
    wsclean_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Extra output logging."
    )
    wsclean_parser.add_argument(
        "--wsclean-container",
        type=Path,
        default=None,
        help="Path to a singularity container with wsclean installed. ",
    )
    wsclean_parser.add_argument(
        "--data-column",
        type=str,
        default="CORRECTED_DATA",
        help="The column name to image. ",
    )

    return parser


def cli() -> None:
    parser = get_parser()

    args = parser.parse_args()

    if args.mode == "image":
        if args.verbose:
            import logging

            logger.setLevel(logging.DEBUG)

        ms = MS(path=args.ms, column=args.data_column)

        wsclean_imager(ms=ms, wsclean_container=args.wsclean_container)


if __name__ == "__main__":
    cli()
