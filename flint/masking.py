"""Utility functions to make image based masks from images, with the initial
thought being towards FITS images. 
"""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from flint.logging import logger


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Simple utility functions to create masks from FITS images. "
    )


def cli():
    parser = get_parser()

    subparser = parser.add_subparsers(
        dest="mode", help="Operation mode of flint_bandpass"
    )

    fits_parser = subparser(
        "fitsmask",
        help="Create a mask for an image, using its RMS and BKG images (e.g. outputs from BANE). Output FITS image will default to the image with a mask suffix.",
    )
    fits_parser.add_argument("image", type=Path, help="Path to the input image. ")
    fits_parser.add_argument(
        "rms", type=Path, help="Path to the RMS of the input image. "
    )
    fits_parser.add_argument(
        "bkg", type=Path, help="Path to the BKG of the input image. "
    )

    fits_parser.add_argument(
        "-s",
        "--save-signal",
        action="store_true",
        help="Save the signal map. Defaults to the same as image with a signal suffix. ",
    )


if __name__ == "__main__":
    cli()
