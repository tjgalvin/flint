"""A basic interface into aegean source finding routines.
"""
from argparse import ArgumentParser
from pathlib import Path
from typing import NamedTuple, Tuple

from AegeanTools import BANE
from AegeanTools.catalogs import save_catalog
from AegeanTools.source_finder import SourceFinder
from astropy.io import fits

from flint.logging import logger
from flint.naming import create_aegean_names
from flint.sclient import run_singularity_command


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


def run_bane_and_aegean(
    image: Path, aegean_container: Path, cores: int = 8
) -> AegeanOutputs:
    """Run BANE, the background and noise estimator, and aegean, the source finder,
    against an input image. This function attempts to hook into the AegeanTools
    module directly, which does not work with dask daemon processes.

    Args:
        image (Path): The input image that BANE will calculate a background and RMS map for
        aegean_container (Path): Path to a singularity container that was the AegeanTools packages installed.
        cores (int, optional): The number of cores to allow BANE to use. Internally BANE will create a number of sub-processes. Defaults to 8.

    Returns:
        AegeanOutputs: The newly created BANE products
    """
    image = image.absolute()
    base_output = str(image.parent / image.stem)
    logger.info(f"Using base output name of: {base_output}")

    aegean_names = create_aegean_names(base_output=base_output)
    logger.debug(f"{aegean_names=}")

    bane_command_str = f"BANE {str(image)} --cores {cores} --stripes {cores//2}"
    logger.info("Constructed BANE command. ")

    bind_dir = [image.absolute()]
    run_singularity_command(
        image=aegean_container, command=bane_command_str, bind_dirs=bind_dir
    )

    # NOTE: Aegean will add the '_comp' component to the output tables. So, if we want
    # basename_comp.fits
    # to be the output component table then we want to pass
    # --table basename.fits
    # and have to rely on aegean doing the right thing.
    aegean_command = f"aegean {str(image)} --autoload --nocov --maxsummits 4 --table {base_output}.fits"
    logger.info("Constructed aegean command. ")
    logger.debug(f"{aegean_command=}")

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
    )

    logger.info(f"Aegeam finished running. {aegean_outputs=}")

    return aegean_outputs


def python_run_bane_and_aegean(image: Path, cores: int = 8) -> AegeanOutputs:
    """Run BANE, the background and noise estimator, and aegean, the source finder,
    against an input image. This function attempts to hook into the AegeanTools
    module directly, which does not work with dask daemon processes.

    Args:
        image (Path): The input image that BANE will calculate a background and RMS map for
        cores (int, optional): The number of cores to allow BANE to use. Internally BANE will create a number of sub-processes. Defaults to 8.

    Returns:
        AegeanOutputs: The newly created BANE products
    """
    base_output = str(image.stem)
    logger.info(f"Using base output name of: {base_output}")

    aegean_names = create_aegean_names(base_output=base_output)

    # Note the cores and slices below. In BANE 2.3.0 there
    # was a bug that could get into a deadlock when attempting
    # to multi-process. Explcitly setting cores to be more
    # than nslices resolves.
    BANE.filter_image(
        im_name=str(image), out_base=base_output, cores=cores, nslice=cores - 3
    )
    # These are the bane outputs
    bkg_image_path = aegean_names.bkg_image
    rms_image_path = aegean_names.rms_image

    logger.info("Have finished running BANE. ")
    assert (
        bkg_image_path.exists()
    ), f"BANE output image {bkg_image_path} does not exists. "
    assert (
        rms_image_path.exists()
    ), f"BANE output image {rms_image_path} does not exists. "

    # TODO: These options need to have an associated class
    logger.info("About to run aegean. ")
    source_finder = SourceFinder()
    _ = source_finder.find_sources_in_image(
        filename=str(image),
        hdu_index=0,
        cube_index=0,
        max_summits=10,
        innerclip=5,
        outerclip=3,
        rmsin=str(rms_image_path),
        bkgin=str(bkg_image_path),
    )

    save_catalog(
        filename=str(aegean_names.comp_cat),
        catalog=source_finder.sources,
    )

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
