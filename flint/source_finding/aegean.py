"""A basic interface into aegean source finding routines."""

from argparse import ArgumentParser
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

from astropy.io import fits

from flint.logging import logger
from flint.naming import create_aegean_names
from flint.sclient import run_singularity_command


class BANEOptions(NamedTuple):
    """Container for basic BANE related options. Only a subclass of BANE options are supported."""

    grid_size: Optional[Tuple[int, int]] = (16, 16)
    """The step interval of each box, in pixels"""
    box_size: Optional[Tuple[int, int]] = (196, 196)
    """The size of the box in pixels"""


class AegeanOptions(NamedTuple):
    """Container for basic aegean options. Only a subclass of aegean options are supported.

    Of note is the lack of a tables option (corresponding to --tables). This is dependent on knowing the base output name
    and relying on aegean to also append a suffic of sorts to the outputs. For that reason
    the aegean command generated will always create the table option.
    """

    nocov: bool = True
    """Whether aegean should attempt to model the co-variance of pixels. If true aegean does not. """
    maxsummits: int = 4
    """The maximum number of components an island is allowed to have before it is ignored. """
    autoload: bool = True
    """Attempt to load precomputed background and rms maps. """


class AegeanOutputs(NamedTuple):
    """Somple structure to represent output aegean products"""

    bkg: Path
    """Background map created by BANE"""
    rms: Path
    """RMS map created by BANE"""
    comp: Path
    """Source component catalogue created by Aegean"""
    beam_shape: Tuple[float, float, float]
    """The `BMAJ`, `BMIN` and `BPA` that were stored in the image header that Aegen searched"""
    image: Path
    """The input image that was used to source find against"""


def _get_bane_command(image: Path, cores: int, bane_options: BANEOptions) -> str:
    """Create the BANE command to run"""
    # The stripes is purposely set lower than the cores due to an outstanding bane bug that can cause a deadlock.
    bane_command_str = f"BANE {str(image)} --cores {cores} --stripes {cores-1} "
    if bane_options.grid_size:
        bane_command_str += (
            f"--grid {bane_options.grid_size[0]} {bane_options.grid_size[1]} "
        )
    if bane_options.box_size:
        bane_command_str += (
            f"--box {bane_options.box_size[0]} {bane_options.box_size[1]}"
        )
    bane_command_str = bane_command_str.rstrip()
    logger.info("Constructed bane command.")

    return bane_command_str


def _get_aegean_command(
    image: Path, base_output: str, aegean_options: AegeanOptions
) -> str:
    """Create the aegean command to run"""
    aegean_command = f"aegean {str(image)} "
    if aegean_options.autoload:
        aegean_command += "--autoload "
    if aegean_options.nocov:
        aegean_command += "--nocov "

    # NOTE: Aegean will add the '_comp' component to the output tables. So, if we want
    # basename_comp.fits
    # to be the output component table then we want to pass
    # --table basename.fits
    # and have to rely on aegean doing the right thing.
    aegean_command += (
        f"--maxsummits {aegean_options.maxsummits} --table {base_output}.fits"
    )

    logger.info("Constructed aegean command. ")
    logger.debug(f"{aegean_command=}")

    return aegean_command


def run_bane_and_aegean(
    image: Path,
    aegean_container: Path,
    cores: int = 8,
    bane_options: Optional[BANEOptions] = None,
    aegean_options: Optional[AegeanOptions] = None,
) -> AegeanOutputs:
    """Run BANE, the background and noise estimator, and aegean, the source finder,
    against an input image. This function attempts to hook into the AegeanTools
    module directly, which does not work with dask daemon processes.

    Args:
        image (Path): The input image that BANE will calculate a background and RMS map for
        aegean_container (Path): Path to a singularity container that was the AegeanTools packages installed.
        cores (int, optional): The number of cores to allow BANE to use. Internally BANE will create a number of sub-processes. Defaults to 8.
        bane_options (Optional[BANEOptions], optional): The options that are provided to BANE. If None defaults of BANEOptions are used. Defaults to None.
        aegean_options (Optional[AegeanOptions], optional): The options that are provided to Aegean. if None defaults of AegeanOptions are used. Defaults to None.

    Returns:
        AegeanOutputs: The newly created BANE products
    """
    bane_options = bane_options if bane_options else BANEOptions()
    aegean_options = aegean_options if aegean_options else AegeanOptions()

    image = image.absolute()
    base_output = str(image.parent / image.stem)
    logger.info(f"Using base output name of: {base_output}")

    aegean_names = create_aegean_names(base_output=base_output)
    logger.debug(f"{aegean_names=}")

    bane_command_str = _get_bane_command(
        image=image, cores=cores, bane_options=bane_options
    )

    bind_dir = [image.absolute().parent]
    run_singularity_command(
        image=aegean_container, command=bane_command_str, bind_dirs=bind_dir
    )

    aegean_command = _get_aegean_command(
        image=image, base_output=base_output, aegean_options=aegean_options
    )
    run_singularity_command(
        image=aegean_container, command=aegean_command, bind_dirs=bind_dir
    )

    # These are the bane outputs
    bkg_image_path = aegean_names.bkg_image
    rms_image_path = aegean_names.rms_image

    image_header = fits.getheader(image)
    image_beam = (
        image_header["BMAJ"],
        image_header["BMIN"],
        image_header["BPA"],
    )

    aegean_outputs = AegeanOutputs(
        bkg=bkg_image_path,
        rms=rms_image_path,
        comp=aegean_names.comp_cat,
        beam_shape=image_beam,
        image=image,
    )

    logger.info(f"Aegeam finished running. {aegean_outputs=}")

    return aegean_outputs


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode")

    bane_parser = subparsers.add_parser(
        name="find", help="Run BANE with default options. "
    )

    bane_parser.add_argument(
        "image", type=Path, help="The image that BANE will process"
    )
    bane_parser.add_argument(
        "container", type=Path, help="Path to container with AegeanTools"
    )
    bane_parser.add_argument(
        "--cores", type=int, default=8, help="Number of cores to instruct aegean to use"
    )

    return parser


def cli() -> None:
    parser = get_parser()

    args = parser.parse_args()

    if args.mode == "find":
        run_bane_and_aegean(
            image=args.image, aegean_container=args.container, cores=args.cores
        )


if __name__ == "__main__":
    cli()
