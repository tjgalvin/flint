"""Simple interface into wsclean
"""
from __future__ import annotations
from pathlib import Path
from typing import NamedTuple, Collection, Union, List, Tuple, Any, Optional, Dict
from numbers import Number
from argparse import ArgumentParser

from flint.logging import logger
from flint.ms import MS
from flint.sclient import run_singularity_command


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
    multiscale: bool = True
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
    ms: Union[MS, Collection[MS]]
    """The measurement sets that have been included in the wsclean command. """


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
    wsclean_cmd = WSCleanCMD(cmd=cmd, ms=ms.with_options(model_column="MODEL_DATA"))

    if container:
        bind_dirs = [ms.path.parent.absolute()]
        run_singularity_command(
            image=container, command=wsclean_cmd.cmd, bind_dirs=bind_dirs
        )

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
