"""Utility functions to make image based masks from images, with the initial
thought being towards FITS images. 
"""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from astropy.io import fits

from flint.logging import logger
from flint.naming import FITSMaskNames, create_fits_mask_names


def create_snr_mask_from_fits(
    fits_image_path: Path,
    fits_rms_path: Path,
    fits_bkg_path: Path,
    create_signal_fits: bool = False,
    min_snr: float = 4.0,
) -> FITSMaskNames:
    """Create a mask for an input FITS image based on a signal to noise given a corresponding pair of RMS and background FITS images.

    Internally the signal image is computed as something akin to:
    > signal = (image - background) / rms

    This is done in a staged manner to minimise the number of (potentially large) images
    held in memory.

    Each of the input images needs to share the same shape. This means that compression
    features offered by some tooling (e.g. BANE --compress) can not be used.

    Once the signal map as been computed, all pixels below ``min_snr`` are flagged.

    Args:
        fits_image_path (Path): Path to the FITS file containing an image
        fits_rms_path (Path): Path to the FITS file with an RMS image corresponding to ``fits_image_path``
        fits_bkg_path (Path): Path to the FITS file with an baclground image corresponding to ``fits_image_path``
        create_signal_fits (bool, optional): Create an output signal map. Defaults to False.
        min_snr (float, optional): Minimum signal-to-noise ratio for the masking to include a pixel. Defaults to 4.0.

    Returns:
        FITSMaskNames: Container describing the signal and mask FITS image paths. If ``create_signal_path`` is None, then the ``signal_fits`` attribute will be None.
    """
    logger.info(f"Creating a mask image with SNR>{min_snr:.2f}")
    mask_names = create_fits_mask_names(
        fits_image=fits_image_path, include_signal_path=create_signal_fits
    )

    with fits.open(fits_image_path) as fits_image:
        fits_header = fits_image[0].header
        with fits.open(fits_bkg_path) as fits_bkg:
            logger.info("Subtracting background")
            signal_data = fits_image[0].data - fits_bkg[0].data

    with fits.open(fits_rms_path) as fits_rms:
        logger.info("Dividing by RMS")
        signal_data /= fits_rms[0].data

    if create_signal_fits:
        logger.info(f"Writing {mask_names.signal_fits}")
        fits.writeto(
            filename=mask_names.signal_fits, data=signal_data, header=fits_header
        )

    mask_data = (signal_data > min_snr).astype(int)

    logger.info(f"Writing {mask_names.mask_fits}")
    fits.writeto(filename=mask_names.mask_fits, data=mask_data, header=fits_header)

    return mask_names


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Simple utility functions to create masks from FITS images. "
    )

    subparser = parser.add_subparsers(
        dest="mode", help="Operation mode of flint_bandpass"
    )

    fits_parser = subparser(
        "snrsmask",
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
    fits_parser.add_argument(
        "--min-snr",
        type=float,
        default=4,
        help="The minimum SNR required for a pixel to be marked as valid. ",
    )

    return parser


def cli():
    parser = get_parser()

    args = parser.parse_args()

    if args.mode == "snrmask":
        create_snr_mask_from_fits(
            fits_image_path=args.image,
            fits_rms_path=args.rms,
            fits_bkg_path=args.bkg,
            create_signal_fits=args.save_signal,
            min_snr=args.min_snr,
        )
    else:
        logger.error(f"Supplied mode {args.mode} is not known. ")


if __name__ == "__main__":
    cli()
