"""A basic interface into aegean source finding routines. 
"""
from argparse import ArgumentParser
from pathlib import Path
from typing import NamedTuple

from AegeanTools import BANE

from flint.logging import logger


class AegeanOutputs(NamedTuple):
    bkg: Path
    rms: Path


def run_bane(image: Path, cores: int = 8) -> AegeanOutputs:
    """Run BANE, the aegean background and noise estimator,
    against an input image.

    Args:
        image (Path): The input image that BANE will calculate a background and RMS map for
        cores (int, optional): The number of cores to allow BANE to use. Internally BANE will create a number of sub-processes. Defaults to 8.

    Returns:
        AegeanOutputs: The newly created BANE products
    """
    base_output = str(image.stem)
    logger.info(f"Using base output name of: {base_output}")

    # Note the cores and slices below. In BANE 2.3.0 there
    # was a bug that could get into a deadlock when attempting
    # to multi-process. Explcitly setting cores to be more
    # than nslices resolves.
    BANE.filter_image(
        im_name=str(image), out_base=base_output, cores=cores, nslice=cores - 1
    )
    # These are the bane outputs
    bkg_image_path = Path(f"{base_output}_bkg.fits")
    rms_image_path = Path(f"{base_output}_rms.fits")

    logger.info(f"Have finished running BANE. ")
    assert (
        not bkg_image_path.exists()
    ), f"BANE output image {bkg_image_path} does not exists. "
    assert (
        not rms_image_path.exists()
    ), f"BANE output image {rms_image_path} does not exists. "

    aegean_outputs = AegeanOutputs(bkg=bkg_image_path, rms=rms_image_path)

    return aegean_outputs


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode")

    bane_parser = subparsers.add_parser(
        name="bane", help="Run BANE with default options. "
    )

    bane_parser.add_argument(
        "image", type=Path, help="The image that BANE will process"
    )

    return parser


def cli() -> None:
    parser = get_parser()

    args = parser.parse_args()

    if args.mode == "bane":
        run_bane(image=args.image)


if __name__ == "__main__":
    cli()
