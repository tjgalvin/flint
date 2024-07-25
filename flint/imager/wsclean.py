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
from flint.naming import create_imaging_name_prefix
from flint.sclient import run_singularity_command
from flint.utils import get_environment_variable, hold_then_move_into


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

    Basic support for environment variables is available. Should a value start
    with `$` it is assumed to be a environment variable, it is will be looked up.
    Some basic attempts to deterimine if it is a path is made.

    Should the `temp_dir` options be specified then all images will be
    created in this location, and then moved over to the same parent directory
    as the imaged MS. This is done by setting the wsclean `-name` argument.
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
    temp_dir: Optional[Union[str, Path]] = None
    """The path to a temporary directory where files will be wrritten. """
    pol: str = "i"
    """The polarisation to be imaged"""
    
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
        prefix (str): The prefix of the imaging run (akin to -name option in wsclean call)
        subbands (int): Number of subbands that were imaged
        pols (Optional[Union[str,Collection[str]]], optional): The polarisation of the image. If None are provided then this is not used. Multiple polarisation may be supplied. If multiple pols are given in an iterable, each will be produced. Defaults to None.
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


def create_wsclean_name_argument(wsclean_options: WSCleanOptions, ms: MS) -> Path:
    """Create the value that will be provided to wsclean -name argument. This has
    to be generated. Among things to consider is the desired output directory of imaging
    files. This by default will be alongside the measurement set. If a `temp_dir` 
    has been specified then output files will be written here. 

    Args:
        wsclean_options (WSCleanOptions): Set of wsclean options to consider
        ms (MS): The measurement set to be imaged

    Returns:
        Path: Value of the -name argument to provide to wsclean
    """
    wsclean_options_dict = wsclean_options._asdict()

    # Prepare the name for the output wsclean command
    # Construct the name property of the string
    pol = wsclean_options_dict["pol"]
    name_prefix_str = create_imaging_name_prefix(ms=ms, pol=pol)
    
    # Now resolve the directory part
    name_dir = ms.path.parent
    temp_dir = wsclean_options_dict.get("temp_dir", None)
    if temp_dir:
        # Resolve if environment variable
        name_dir = (
            get_environment_variable(variable=temp_dir)
            if isinstance(temp_dir, str) and temp_dir[0] == "$"
            else Path(temp_dir)
        )
        assert name_dir is not None, f"{name_dir=} is None, which is bad"
        
    name_argument_path = Path(name_dir) / name_prefix_str
    logger.info(f"Constructed -name {name_argument_path}")

    return name_argument_path

def create_wsclean_cmd(
    ms: MS, wsclean_options: WSCleanOptions, container: Optional[Path] = None
) -> WSCleanCommand:
    """Create a wsclean command from a WSCleanOptions container

    For the most part these are one-to-one mappings to the wsclean CLI with the
    exceptions being:
    #. the `-name` argument will be generated and supplied to the CLI string and will default to the parent directory and name of the supplied measurement set
    #. If `wsclean_options.temp_dir` is specified this directory is used in place of the measurement sets parent directory

    If `container` is supplied to immediatedly execute this command then the 
    output wsclean image products will be moved from the `temp-dir` to the
    same directory as the measurement set. 

    Args:
        ms (MS): The measurement set to be imaged
        wsclean_options (WSCleanOptions): WSClean options to image with
        container (Optional[Path], optional): If a path to a container is provided the command is executed immediatedly. Defaults to None.

    Raises:
        ValueError: Raised when a option has not been successfully processed

    Returns:
        WSCleanCommand: The wsclean command to run
    """
    # TODO: This is very very smelly. I think removing the `.name` from WSCleanOptions
    # to start with, build that as an explicit testable function, and pass/return the name
    # argument alongside the prefix in the WSCleanCMD. Also need to rename that, its a horrible
    # name for a variable and ship
    
    # Some options should also extend the singularity bind directories
    bind_dir_paths = []
    bind_dir_options = ("temp-dir",)

    # Some wsclean options, if multiple values are provided, might need
    # to be join as a csv list. Others might want to be dumped in. Just
    # attempting to future proof (arguably needlessly).
    options_to_comma_join = "multiscale-scales"

    name_argument_path = create_wsclean_name_argument(wsclean_options=wsclean_options, ms=ms)
    move_directory = ms.path.parent
    hold_directory: Optional[Path] = Path(name_argument_path).parent

    wsclean_options_dict = wsclean_options._asdict()

    cmd = ["wsclean"]
    unknowns: List[Tuple[Any, Any]] = []
    logger.info("Creating wsclean command.")

    # TODO: The inside of this for loop should be expressed as a `map`. 
    # Put the contents of the if conditionals into a separate function and add tests
    for key, value in wsclean_options_dict.items():
        key = key.replace("_", "-")
        logger.debug(f"{key=} {value=} {type(value)=}")

        value = (
            get_environment_variable(variable=value)
            if isinstance(value, str) and value[0] == "$"
            else value
        )

        if key == "size":
            cmd += [f"-size {value} {value}"]
        elif key == "wgridder-accuracy":
            if wsclean_options.gridder == "wgridder":
                cmd += [f"-{key} {value}"]
        elif isinstance(value, bool):
            if value:
                cmd += [f"-{key}"]
        elif isinstance(value, (str, Number)):
            cmd += [f"-{key} {value}"]
        elif isinstance(value, (list, tuple)):
            value = list(map(str, value))
            value_str = (
                ",".join(value) if key in options_to_comma_join else " ".join(value)
            )
            cmd += [f"-{key} {value_str}"]
        elif isinstance(value, Path):
            value_str = str(value)
            cmd += [f"-{key} {value_str}"]
        elif value is None:
            logger.debug(
                f"{key} option set to {value}. Not sure what this means. Ignoring. "
            )
        else:
            unknowns.append((key, value))

        if key in bind_dir_options and isinstance(value, (str, Path)):
            bind_dir_paths.append(Path(value))

    if len(unknowns) > 0:
        msg = ", ".join([f"{t[0]} {t[1]}" for t in unknowns])
        raise ValueError(f"Unknown wsclean option types: {msg}")

    cmd += [f"-name {str(name_argument_path)}"]
    cmd += [f"{str(ms.path)} "]

    cmd = " ".join(cmd)

    logger.info(f"Constructed wsclean command: {cmd=}")
    logger.info("Setting default model data column to 'MODEL_DATA'")
    wsclean_cmd = WSCleanCommand(
        cmd=cmd, options=wsclean_options, ms=ms.with_options(model_column="MODEL_DATA")
    )

    if container:
        wsclean_cmd = run_wsclean_imager(
            wsclean_cmd=wsclean_cmd,
            container=container,
            bind_dirs=tuple(bind_dir_paths),
            move_hold_directories=(move_directory, hold_directory),
               image_prefix_str=str(name_argument_path),
        )

    return wsclean_cmd


def run_wsclean_imager(
    wsclean_cmd: WSCleanCommand,
    container: Path,
    bind_dirs: Optional[Tuple[Path]] = None,
    move_hold_directories: Optional[Tuple[Path, Optional[Path]]] = None,
    image_prefix_str: Optional[str] = None,
) -> WSCleanCommand:
    """Run a provided wsclean command. Optionally will clean up files,
    including the dirty beams, psfs and other assorted things.

    An `ImageSet` is constructed that attempts to capture the output wsclean image products. If `image_prefix_str`
    is specified the image set will be created by (ordered by preference):
    #. Adding the `image_prefix_str` to the `move_directory`
    #. Guessing it from the path of the measurement set from `wsclean_cmd.ms.path`

    Args:
        wsclean_cmd (WSCleanCommand): The command to run, and other properties (cleanup.)
        container (Path): Path to the container with wsclean available in it
        bind_dirs (Optional[Tuple[Path]], optional): Additional directories to include when binding to the wsclean container. Defaults to None.
        move_hold_directories (Optional[Tuple[Path,Optional[Path]]], optional): The `move_directory` and `hold_directory` passed to the temporary context manger. If None no `hold_then_move_into` manager is used. Defaults to None.
        image_prefix_str (Optional[str], optional): The name used to search for wsclean outputs. If None, it is guessed from the name and location of the MS. Defaults to None.

    Returns:
        WSCleanCommand: The executed wsclean command with a populated imageset properter.
    """

    ms = wsclean_cmd.ms
    if isinstance(ms, MS):
        ms = (ms,)

    sclient_bind_dirs = [Path(m.path).parent.absolute() for m in ms]
    if bind_dirs:
        sclient_bind_dirs = sclient_bind_dirs + list(bind_dirs)

    if move_hold_directories:
        with hold_then_move_into(
            move_directory=move_hold_directories[0],
            hold_directory=move_hold_directories[1],
        ) as directory:
            sclient_bind_dirs.append(directory)
            run_singularity_command(
                image=container,
                command=wsclean_cmd.cmd,
                bind_dirs=sclient_bind_dirs,
                stream_callback_func=_wsclean_output_callback,
            )

            # Update the prefix based on where the files will be moved to
            image_prefix_str = (
                f"{str(move_hold_directories[0] / Path(image_prefix_str).name)}"
                if image_prefix_str
                else None
            )
    else:
        run_singularity_command(
            image=container,
            command=wsclean_cmd.cmd,
            bind_dirs=sclient_bind_dirs,
            stream_callback_func=_wsclean_output_callback,
        )

    prefix = image_prefix_str if image_prefix_str else None
    if prefix is None:
        prefix = str(ms[0].path.parent / ms[0].path.name)
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
    update_wsclean_options: Optional[Dict[str, Any]] = None,
) -> WSCleanCommand:
    """Create and run a wsclean imager command against a measurement set.

    Args:
        ms (Union[Path,MS]): Path to the measurement set that will be imaged
        wsclean_container (Path): Path to the container with wsclean installed
        update_wsclean_options (Optional[Dict[str, Any]], optional): Additional options to update the generated WscleanOptions with. Keys should be attributes of WscleanOptions. Defaults ot None.

    Returns:
        WSCleanCommand: _description_
    """

    # TODO: This should be expanded to support multiple measurement sets
    ms = MS.cast(ms)

    wsclean_options = WSCleanOptions()
    if update_wsclean_options:
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
