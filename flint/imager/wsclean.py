"""Simple interface into wsclean
"""
from __future__ import annotations
from glob import glob
from pathlib import Path
from typing import NamedTuple, Collection, Union, List, Tuple, Any, Optional, Dict
from numbers import Number
from argparse import ArgumentParser

from flint.logging import logger
from flint.ms import MS
from flint.sclient import run_singularity_command


class ImageSet(NamedTuple):
    """A structure to represent the images and auxillary products produced by
    wsclean"""

    prefix: str
    """Prefix of the images and other output products. This should correspond to the -name argument from wsclean"""
    images: Dict[str, Collection[Path]]
    """Images produced. The Stokes (e.g. I/Q/U/V) act as the key to the corresponding set of images. """
    psfs: Optional[Collection[Path]] = None
    """References to the PSFs produced by wsclean. """
    dirty_images: Optional[Dict[str, Collection[Path]]] = None
    """Dirty images. The Stokes (e.g. I/Q/U/V) are the key to the corresponding set of images. """
    model_images: Optional[Dict[str, Collection[Path]]] = None
    """Model images. The Stokes (e.g. I/Q/U/V) are the key to the corresponding set of images. """
    residual_images: Optional[Dict[str, Collection[Path]]] = None
    """Residual images. The Stokes (e.g. I/Q/U/V) are the key to the corresponding set of images. """


class WSCleanOptions(NamedTuple):
    """A basic container to handle WSClean options. These attributes should
    conform to the same option name in the calling signature of wsclean
    """

    abs_mem: int = 100
    """Memory wsclean should try to limit itself to"""
    local_rms_window: int = 65
    """Size of the window used to estimate rms noise"""
    size: int = 6000
    """Image size"""
    local_rms: bool = True
    """Whether a local rms map is computed"""
    force_mask_rounds: int = 6
    """Round of force masked derivation"""
    auto_mask: float = 3.5
    """How deep the construct clean mask is during each cycle"""
    auto_threshold: float = 0.5
    """How deep to clean once initial clean threshold reached"""
    channels_out: int = 4
    """Number of output channels"""
    mgain: float = 0.7
    """Major cycle gain"""
    nmiter: int = 15
    """Maximum number of major cycles to perform"""
    niter: int = 100000
    """Maximum numer of minor cycles"""
    multiscale: bool = False
    """Enable multiscale deconvolution"""
    multiscale_scale_bias: float = 0.7
    """Multiscale bias term"""
    fit_spectral_pol: int = 4
    """Number of spectral terms to include during sub-band subtractin"""
    weight: str = "briggs -1.5"
    """Robustness of the weighting used"""
    data_column: str = "CORRECTED_DATA"
    """Which column in the MS to image"""
    scale: str = "2.5asec"
    """Pixel scale size"""
    gridder: str = "wgridder"
    """Use the wgridder kernel in wsclean (instead of the w-stacking method)"""
    join_channels: bool = True
    """Collapse the sub-band images down to an MFS image when peak-finding"""
    name: Optional[str] = None

    def with_options(self, **kwargs) -> WSCleanOptions:
        """Return a new instance of WSCleanOptions with updated components"""
        _dict = self._asdict()
        _dict.update(**kwargs)

        return WSCleanOptions(**_dict)


class WSCleanCMD(NamedTuple):
    """Simple container for a wsclean command."""

    cmd: str
    """The constructede wsclean command that would be executed."""
    options: WSCleanOptions
    """The set of wslean options used for imaging"""
    ms: Union[MS, Collection[MS]]
    """The measurement sets that have been included in the wsclean command. """
    imageset: Optional[ImageSet] = None
    """Collection of images produced by wsclean"""
    cleanup: bool = True
    """Will clean up the dirty images/psfs/residuals/models when the imaging has completed"""


def find_wsclean_images(
    prefix: str, search_dir: Path = Path("."), stokes: str = "I"
) -> ImageSet:
    modes = ("image", "psf", "model", "dirty")

    # In the current form Stokes-I imaging is the only image performed
    i_images = search_dir.glob(f"{prefix}*{{MFS,[0-9][0-9][0-9][0-9]}}-image.fits")

    wsclean_imageset = ImageSet(prefix=prefix, images={"I": i_images})


def get_wsclean_output_names(
    prefix: str,
    subbands: int,
    pols: Optional[Union[str,Collection[str]]] = None,
    verify_exists: bool = False,
    include_mfs: bool = True,
    output_types: Union[str,Collection[str]] = ('image','dirty','residual','model', 'psf')
) -> Dict[str,Collection[Path]]:
    """Attempt to generate the file names and paths that would be
    created by an imaging run of wsclean.

    Args:
        prefix (str): The prefix of the imaging run (the -name option in wsclean call)
        subbands (int): Number of subbands that were imaged
        pol (Optional[Union[str,Collection[str]]], optional): The polarisation of the image. If None are provided then this is not used. Multiple polarisation may be supplied. If multiple pols are given in an iterable, each will be produced. Defaults to None.
        verify_exists (bool, optional): Ensures that each generated path corresponds to an actual file. Defaults to False.
        include_mfs (bool, optional): Include the MFS images produced by wsclean. Defaults to True.
        output_types (Union[str,Collection[str]]): Include files of this type, including image, dirty, residual, model, psf. Defaults to  ('image','dirty','residual','model', 'psf').

    Raises:
        FileExistsError: Raised when a file does not exist and verify_exists is True.

    Returns:
        Dict[str,Collection[Path]]: The file paths that wsclean should create/has created.
    """
    # TODO: NEED TESTS!
    subband_strs = [f"{subband:04}" for subband in range(subbands)]
    if include_mfs:
        subband_strs.append("MFS")

    if pols is None:
        pols = (None, )
    elif isinstance(pols, str):
        pols = (pols, )
    
    if isinstance(output_types, str):
        output_types = (output_types,)
    
    images: Dict[str,Collection[Path]] = {}
    for image_type in ("image", "dirty", "model", "residual"):
        if not image_type in output_types:
            continue
        
        paths: List[Path] = []
        for pol in pols:
            for subband_str in subband_strs:
                if pol:
                    path_str = f"{prefix}-{subband_str}-{pol}-{image_type}.fits"
                else:
                    path_str = f"{prefix}-{subband_str}-{image_type}.fits"

                paths.append(Path(path_str))

        images[image_type] = paths
 
    # The PSF is the same for all stokes
    if 'psf' in output_types:
        images["psf"] = [Path(f"{prefix}-{subband_str}-psf.fits") for subband_str in subband_strs]
    
    if verify_exists:
        paths_no_exists: List[Path] = []
        for _, paths in images.items():
            paths_no_exists += [path for path in paths if not path.exists()]
        if len(paths_no_exists) > 0:
            raise FileExistsError(f"The following {len(paths_no_exists)} files do not exist: {paths_no_exists}")
        
    return images


def delete_wsclean_outputs(prefix: str, output_type: str='image', ignore_mfs: bool=True) -> Collection[Path]:
    """Attempt to remove elected wsclean output files

    Args:
        prefix (str): The prefix of the files to remove. This would correspond to the -name of wsclean. 
        output_type (str, optional): What type of wsclean output to try to remove. Defaults to 'image'.
        ignore_mfs (bool, optional): If True, do not remove MFS outputs (attempt to, atleast). Defaults to True. 

    Returns:
        Collection[Path]: The paths that were removed (or at least attempted to be removed)/
    """
    
    paths = [Path(p) for p in glob(f"{prefix}*{output_type}.fits")]
    logger.info(f"Found {len(paths)} matching {prefix=} and {output_type=}.")
    rm_paths: List[Path] = []

    for path in path:
        if ignore_mfs and '-MFS-' in str(path.name):
            logger.info(f"{path} appears to be an MFS product, not removing. ")
            continue
        if path.exists():
            logger.warn(f"Removing {path}.")
            try:
                rm_paths.append(path)
                path.unlink()
            except Exception as e:
                logger.criticial(f"Removing {path} failed: {e}")

    return rm_paths

def create_wsclean_cmd(
    ms: MS, wsclean_options: WSCleanOptions, container: Optional[Path] = None
) -> WSCleanCMD:
    """Create a wsclean command from a WSCleanOptions container

    Args:
        ms (MS): The measurement set to be imaged
        wsclean_options (WSCleanOptions): WSClean options to image with
        container (Optional[Path], optional): If a path to a container is provided the command is executed immediatedly. Defaults to None.

    Raises:
        ValueError: Raised when a option has not been successfully processed

    Returns:
        WSCleanCMD: The wsclean command to run
    """

    cmd = "wsclean "
    unknowns: List[Tuple[Any, Any]] = []
    logger.info("Creating wsclean command.")
    for key, value in wsclean_options._asdict().items():
        key = key.replace("_", "-")
        logger.debug(f"{key=} {value=} {type(value)=}")
        if key == "size":
            cmd += f"-size {value} {value} "
        elif isinstance(value, bool):
            if value:
                cmd += f"-{key} "
        elif isinstance(value, (str, Number)):
            cmd += f"-{key} {value} "
        else:
            unknowns.append((key, value))

    if len(unknowns) > 0:
        msg = ", ".join([f"{t[0]} {t[1]}" for t in unknowns])
        raise ValueError(f"Unknown wsclean option types: {msg}")

    cmd += f"{str(ms.path)} "

    logger.info(f"Constructed wsclean command: {cmd=}")
    logger.info(f"Setting default model data column to 'MODEL_DATA'")
    wsclean_cmd = WSCleanCMD(
        cmd=cmd, options=wsclean_options, ms=ms.with_options(model_column="MODEL_DATA")
    )

    if container:
        wsclean_cmd = run_wsclean_imager(wsclean_cmd=wsclean_cmd, container=container)

    return wsclean_cmd


def run_wsclean_imager(wsclean_cmd: WSCleanCMD, container: Path) -> WSCleanCMD:
    """Run a provided wsclean command. Optionally will clean up files,
    including the dirty beams, psfs and other assorted things.

    Args:
        wsclean_cmd (WSCleanCMD): The command to run, and other properties (cleanup.)
        container (Path): Path to the container with wsclean available in it

    Returns:
        WSCleanCMD: The executed wsclean command with a populated imageset properter.
    """

    ms = wsclean_cmd.ms
    bind_dirs = [ms.path.parent.absolute()]
    run_singularity_command(
        image=container, command=wsclean_cmd.cmd, bind_dirs=bind_dirs
    )

    prefix = wsclean_cmd.options.name
    if wsclean_cmd.cleanup:
        logger.info(
            f"Will clean up files created by wsclean. "
        )

        for output_type in ("dirty", "psf", "model", "residual"):
            delete_wsclean_outputs(prefix=prefix, output_type=output_type)
    
    images = get_wsclean_output_names(
        prefix=prefix, subbands=wsclean_cmd.options.channels_out, verify_exists=True, output_types='image'
    )
    
    logger.info(f"Found {images=}")
           
    return wsclean_cmd


def wsclean_imager(
    ms: Union[Path, MS],
    wsclean_container: Path,
    wsclean_options_path: Optional[Path] = None,
    update_wsclean_options: Optional[Dict[str, Any]] = None,
) -> WSCleanCMD:
    """Create and run a wsclean imager command against a measurement set.

    Args:
        ms (Union[Path,MS]): Path to the measurement set that will be imaged
        wsclean_container (Path): Path to the container with wsclean installed
        wsclean_options_path (Optional[Path], optional): Location of a wsclean set of options. Defaults to None.

    Returns:
        WSCleanCMD: _description_
    """

    # TODO: This should be expanded to support multiple measurement sets
    ms = MS.cast(ms)

    if wsclean_options_path:
        logger.warn(
            f"This is a place holder for loading a wsclean imager parameter file. It is being ignored. "
        )

    wsclean_options = WSCleanOptions()
    if update_wsclean_options is not None:
        logger.info(f"Updatting wsclean options with user-provided items. ")
        wsclean_options = wsclean_options.with_options(**update_wsclean_options)

    if wsclean_options.name is None:
        # TODO: Come up with a consistent naming scheme. Add in a naming submodule
        # to consolidate this functionality
        wsclean_name = ms.path.absolute().parent / ms.path.stem
        logger.warn(f"Autogenerated wsclean output name: {wsclean_name}")
        wsclean_options = wsclean_options.with_options(name=str(wsclean_name))

    assert ms.column is not None, f"A MS column needs to be elected for imaging. "
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
