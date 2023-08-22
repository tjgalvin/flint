"""A basic interface into aegean source finding routines. 
"""
from argparse import ArgumentParser
from pathlib import Path
from typing import NamedTuple

from AegeanTools import BANE
from AegeanTools.source_finder import SourceFinder
from AegeanTools.catalogs import save_catalog

from flint.logging import logger
from flint.naming import create_aegean_names


class AegeanOutputs(NamedTuple):
    """Somple structure to represent output aegean products"""

    bkg: Path
    """Background map created by BANE"""
    rms: Path
    """RMS map created by BANE"""
    comp: Path
    """Source component catalogue created by Aegean"""


def run_bane_and_aegean(image: Path, cores: int = 8) -> AegeanOutputs:
    """Run BANE, the background and noise estimator, and aegean, the source finder,
    against an input image.

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
        im_name=str(image), out_base=base_output, cores=cores, nslice=cores - 1
    )
    # These are the bane outputs
    bkg_image_path = aegean_names.bkg_image
    rms_image_path = aegean_names.rms_image

    logger.info(f"Have finished running BANE. ")
    assert (
        bkg_image_path.exists()
    ), f"BANE output image {bkg_image_path} does not exists. "
    assert (
        rms_image_path.exists()
    ), f"BANE output image {rms_image_path} does not exists. "

    # TODO: These options need to have an associated class
    logger.info(f"About to run aegean. ")
    source_finder = SourceFinder()
    sf_results = source_finder.find_sources_in_image(
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

    aegean_outputs = AegeanOutputs(
        bkg=bkg_image_path, rms=rms_image_path, comp=aegean_names.comp_cat
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

    return parser


def cli() -> None:
    parser = get_parser()

    args = parser.parse_args()

    if args.mode == "find":
        run_bane_and_aegean(image=args.image)


if __name__ == "__main__":
    cli()
