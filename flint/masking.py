"""Utility functions to make image based masks from images, with the initial
thought being towards FITS images.
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from scipy.ndimage import (
    binary_dilation as scipy_binary_dilation,  # Rename to distinguish from skimage
)
from scipy.ndimage import (
    binary_erosion as scipy_binary_erosion,  # Rename to distinguish from skimage
)
from scipy.ndimage import label

from skimage.filters import butterworth
from skimage.morphology import binary_erosion

from flint.logging import logger
from flint.naming import FITSMaskNames, create_fits_mask_names


def extract_beam_mask_from_mosaic(
    fits_beam_image_path: Path, fits_mosaic_mask_names: FITSMaskNames
) -> FITSMaskNames:
    """Extract a region based on an existing FITS image from a masked FITS
    image. Here a masked FITS image is intended to be one created by
    ``create_snr_mask_from_fits``.

    A simple cutout (e.g. ``CutOut2D``) might not work as intended, as
    when creating the field image (the intended use case here) might be
    regridded onto a unique pixel grid that does not correspond to one
    that would be constructed by wsclean.

    Args:
        fits_beam_image_path (Path): The template image with a valid WCS. This region is used to extract the masked region
        fits_mosaic_mask_names (FITSMaskNames): The previously masked image

    Returns:
        FITSMaskNames: _description_
    """
    # TODO: Ideally we can accept an arbitary WCS, or read the wsclean docs to
    # try to construct it ourselves. The last thing that this pirate wants is
    # to run the imager in a dry-run type mode n cleaning type mode purely for
    # the WCS.

    mask_names = create_fits_mask_names(fits_image=fits_beam_image_path)

    with fits.open(fits_beam_image_path) as beam_image:
        header = beam_image[0].header
        beam_image_shape = beam_image[0].data.shape[-2:]

    logger.info(
        f"Will extract {beam_image_shape} from {fits_mosaic_mask_names.mask_fits}"
    )

    with fits.open(fits_mosaic_mask_names.mask_fits) as mosaic_mask:
        logger.info("Extracting region")
        extract_image = reproject_interp(
            input_data=(
                np.squeeze(mosaic_mask[0].data),
                WCS(mosaic_mask[0].header).celestial,
            ),
            output_projection=WCS(header).celestial,
            shape_out=beam_image_shape,
            order=0,
        )
    logger.info("Clipping extract mask results (interpolation errors)")
    extract_image = np.clip(extract_image, 0.0, 1.0)

    fits.writeto(mask_names.mask_fits, extract_image[0].astype(np.int32), header)

    return mask_names


def _get_signal_image(
    image: Optional[np.ndarray] = None,
    rms: Optional[np.ndarray] = None,
    background: Optional[np.ndarray] = None,
    signal: Optional[np.ndarray] = None,
) -> np.ndarray:
    if all([item is None for item in (image, background, rms, signal)]):
        raise ValueError("No input maps have been provided. ")

    if signal is None:
        if background is None:
            logger.info("No background supplied, assuming zeros. ")
            background = np.zeros_like(image)

        signal = (image - background) / rms

    return signal


def grow_low_snr_mask(
    image: Optional[np.ndarray] = None,
    rms: Optional[np.ndarray] = None,
    background: Optional[np.ndarray] = None,
    signal: Optional[np.ndarray] = None,
    grow_low_snr: float = 2.0,
    grow_low_island_size: int = 512,
    region_mask: Optional[np.ndarray] = None,
) -> np.array:
    """There may be cases where normal thresholding operations based on simple pixel-wise SNR
    cuts fail to pick up diffuse, low surface brightness regions of emission. When some type
    of masking operation is used there may be instances where these regions are never cleaned.

    Sometimes smoothing can help to pick up these features, but when attempting to pass such
    a mask through to the imagery of choice the islands may be larger than the features at their
    native resolution, unless some other more sophisticated filtering is performed.

    This function attempts to grow masks to capture islands of contigous pixels above a low
    SNR cut that would otherwise go uncleaned.

    Args:
            Args:
        image (Optional[np.ndarray], optional): The total intensity pixels to have the mask for. Defaults to None.
        rms (Optional[np.ndarray], optional): The noise across the image. Defaults to None.
        background (Optional[np.ndarray], optional): The background acros the image. If None, zeros are assumed. Defaults to None.
        signal(Optional[np.ndarray], optional): A signal map. Defaults to None.
        grow_low_snr (float, optional): The SNR pixekls have to be above. Defaults to 2.
        grow_low_island_size (int, optional): The minimum number of pixels an island should be for it to be considered valid. Defaults to 512.
        region_mask (Optional[np.ndarray], optional): Whether some region should be masked out before the island size constraint is applied. Defaults to None.

    Returns:
        np.array: The derived mask of objects with low-surface brightness
    """
    # TODO: The `grow_low_island_size` should be represented in solid angle relative to the restoring beam

    signal = _get_signal_image(
        image=image, rms=rms, background=background, signal=signal
    )

    logger.info(
        f"Growing mask for low surface brightness using {grow_low_snr=} {grow_low_island_size=}"
    )
    low_snr_mask = signal > grow_low_snr
    low_snr_mask = scipy_binary_dilation(
        input=low_snr_mask, iterations=2, structure=np.ones((3, 3))
    )
    low_snr_mask = scipy_binary_erosion(
        input=low_snr_mask, iterations=2, structure=np.ones((3, 3))
    )

    if region_mask is not None:
        low_snr_mask[region_mask] = False

    mask_labels, no_labels = label(low_snr_mask, structure=np.ones((3, 3)))
    _, counts = np.unique(mask_labels.flatten(), return_counts=True)

    small_islands = [idx for idx, count in enumerate(counts) if count < 512 and idx > 0]
    low_snr_mask[np.isin(mask_labels, small_islands)] = False

    return low_snr_mask


def reverse_negative_flood_fill(
    image: Optional[np.ndarray] = None,
    rms: Optional[np.ndarray] = None,
    background: Optional[np.ndarray] = None,
    signal: Optional[np.ndarray] = None,
    positive_seed_clip: float = 4,
    positive_flood_clip: float = 2,
    negative_seed_clip: Optional[float] = 5,
    guard_negative_dilation: float = 50,
    grow_low_snr: Optional[float] = 2,
    grow_low_island_size: int = 512,
) -> np.ndarray:
    """Attempt to:

    * seed masks around bright regions of an image and grow them to lower significance thresholds
    * remove regions of negative and positive islands that surrond bright sources.

    An initial set of islands (and masks) are constructed by first
    using the `positive_seed_clip` to create an initial SNR based
    mask. These islands then are binary dilated to grow the islands
    to adjacent pixels at a lower signifcance level (see `scipy.ndimage.binary_dilation`).

    Next an attempt is made to remove artefacts around bright sources,  where
    there are likely to be positive and negative artefacts
    that originate from calibration errors, deconvolution errors, or residual
    structure from an incomplete clean.

    This operation will search for islands of _negative_ pixels above a
    threshold. These pixels are then grown after a guard mask has been constructed
    around bright pixels.

    The assumptions that go into this process:

    * the no genuine source of negative sky emission
    * if there are bright negative artefacts there are likely bright positive artefacts nearby
    * such negative pixels are ~10% level artefacts from a presumed bright sources

    Optionally, the `growlow_snr_mask` may also be considered via the `grow_low_snr` and `grow_low_island_size`
    parameters.

    Args:
        image (Optional[np.ndarray], optional): The total intensity pixels to have the mask for. Defaults to None.
        rms (Optional[np.ndarray], optional): The noise across the image. Defaults to None.
        background (Optional[np.ndarray], optional): The background acros the image. If None, zeros are assumed. Defaults to None.
        signal(Optional[np.ndarray], optional): A signal map. Defaults to None.
        positive_seed_clip (float, optional): Initial clip of the mask before islands are grown. Defaults to 4.
        positive_flood_clip (float, optional): Pixels above `positive_seed_clip` are dilated to this threshold. Defaults to 2.
        negative_seed_clip (Optional[float], optional): Initial clip of negative pixels. This operation is on the inverted signal mask (so this value should be a positive number). If None this second operation is not performed. Defaults to 5.
        guard_negative_dilation (float, optional): Positive pixels from the computed signal mask will be above this threshold to be protect from the negative island mask dilation. Defaults to 50.
        grow_low__snr (Optional[float], optional): Attempt to grow islands of contigous pixels above thius low SNR ration. If None this is not performed. Defaults to 2.
        grow_low_island_size (int, optional): The number of pixels a low SNR should be in order to be considered valid. Defaults to 512.

    Returns:
        np.ndarray: Mask of the pixels to clean
    """

    logger.info("Will be reversing flood filling")
    logger.info(f"{positive_seed_clip=} ")
    logger.info(f"{positive_flood_clip=} ")
    logger.info(f"{negative_seed_clip=} ")
    logger.info(f"{guard_negative_dilation=}")

    signal = _get_signal_image(
        image=image, rms=rms, background=background, signal=signal
    )

    # This Pirate thinks provided the background is handled
    # that taking the inverse is correct
    negative_signal = -1 * signal

    # Here we create the mask image that will start the binary dilation
    # process, and we will ensure only pixels above the `positive_flood_clip`
    # are allowed to be dilated. In other words we are growing the mask
    positive_mask = signal >= positive_seed_clip
    positive_dilated_mask = scipy_binary_dilation(
        input=positive_mask,
        mask=signal > positive_flood_clip,
        iterations=1000,
        structure=np.ones((3, 3)),
    )

    # Now do the same but on negative islands. The assumption here is that:
    # - no genuine source of negative sky emission
    # - negative islands are around bright sources with deconvolution/calibration errors
    # - if there are brightish negative islands there is also positive brightish arteefact islands nearby
    # For this reason the guard mask should be sufficently high to protect the
    # main source but nuke the fask positive islands
    negative_dilated_mask = None
    if negative_seed_clip:
        negative_mask = negative_signal > negative_seed_clip
        negative_dilated_mask = scipy_binary_dilation(
            input=negative_mask,
            mask=signal < guard_negative_dilation,
            iterations=10,
            structure=np.ones((3, 3)),
        )

        # and here we set the presumable nasty islands to False
        positive_dilated_mask[negative_dilated_mask] = False

    if grow_low_snr:
        low_snr_mask = grow_low_snr_mask(
            signal=signal,
            grow_low_snr=grow_low_snr,
            grow_low_island_size=grow_low_island_size,
            region_mask=negative_dilated_mask,
        )
        positive_dilated_mask[low_snr_mask] = True

    return positive_dilated_mask.astype(np.int32)


def create_snr_mask_wbutter_from_fits(
    fits_image_path: Path,
    fits_rms_path: Path,
    fits_bkg_path: Path,
    create_signal_fits: bool = False,
    min_snr: float = 5,
    connectivity_shape: Tuple[int, int] = (4, 4),
    overwrite: bool = True,
) -> FITSMaskNames:
    """Create a mask for an input FITS image based on a signal to noise given a corresponding pair of RMS and background FITS images.

    Internally the signal image is computed as something akin to:
    > signal = (image - background) / rms

    Before deriving a signal map the image is first smoothed using a butterworth filter, and a
    crude rescaling factor is applied based on the ratio of the maximum pixel values before and
    after the smoothing is applied.

    This is done in a staged manner to minimise the number of (potentially large) images
    held in memory.

    Each of the input images needs to share the same shape. This means that compression
    features offered by some tooling (e.g. BANE --compress) can not be used.

    Once the signal map as been computed, all pixels below ``min_snr`` are flagged. The resulting
    islands then have a binary erosion applied to contract the resultingn islands.

    Args:
        fits_image_path (Path): Path to the FITS file containing an image
        fits_rms_path (Path): Path to the FITS file with an RMS image corresponding to ``fits_image_path``
        fits_bkg_path (Path): Path to the FITS file with an baclground image corresponding to ``fits_image_path``
        create_signal_fits (bool, optional): Create an output signal map. Defaults to False.
        min_snr (float, optional): Minimum signal-to-noise ratio for the masking to include a pixel. Defaults to 3.5.
        connectivity_shape (Tuple[int, int], optional): The connectivity matrix used in the scikit-image binary erosion applied to the mask. Defaults to (4, 4).
        overwrite (bool): Passed to `fits.writeto`, and will overwrite files should they exist. Defaults to True.

    Returns:
        FITSMaskNames: Container describing the signal and mask FITS image paths. If ``create_signal_path`` is None, then the ``signal_fits`` attribute will be None.
    """
    logger.info(f"Creating a mask image with SNR>{min_snr:.2f}")
    mask_names = create_fits_mask_names(
        fits_image=fits_image_path, include_signal_path=create_signal_fits
    )

    with fits.open(fits_image_path) as fits_image:
        fits_header = fits_image[0].header

        image_max = np.nanmax(fits_image[0].data)
        image_butter = butterworth(
            np.nan_to_num(np.squeeze(fits_image[0].data)), 0.045, high_pass=False
        )

        butter_max = np.nanmax(image_butter)
        scale_ratio = image_max / butter_max
        logger.info(f"Scaling smoothed image by {scale_ratio:.4f}")
        image_butter *= image_max / butter_max

    with fits.open(fits_bkg_path) as fits_bkg:
        logger.info("Subtracting background")
        signal_data = image_butter - np.squeeze(fits_bkg[0].data)

    with fits.open(fits_rms_path) as fits_rms:
        logger.info("Dividing by RMS")
        signal_data /= np.squeeze(fits_rms[0].data)

    if create_signal_fits:
        logger.info(f"Writing {mask_names.signal_fits}")
        fits.writeto(
            filename=mask_names.signal_fits,
            data=signal_data,
            header=fits_header,
            overwrite=overwrite,
        )

    # Following the help in wsclean:
    # WSClean accepts masks in CASA format and in fits file format. A mask is a
    # normal, single polarization image file, where all zero values are interpreted
    # as being not masked, and all non-zero values are interpreted as masked. In the
    # case of a fits file, the file may either contain a single frequency or it may
    # contain a cube of images.
    logger.info(f"Clipping using a {min_snr=}")
    mask_data = (signal_data > min_snr).astype(np.int32)

    logger.info(f"Applying binary erosion with {connectivity_shape=}")
    mask_data = binary_erosion(mask_data, np.ones(connectivity_shape))

    logger.info(f"Writing {mask_names.mask_fits}")
    fits.writeto(
        filename=mask_names.mask_fits,
        data=mask_data.astype(np.int32),
        header=fits_header,
        overwrite=overwrite,
    )

    return mask_names


def create_snr_mask_from_fits(
    fits_image_path: Path,
    fits_rms_path: Path,
    fits_bkg_path: Path,
    create_signal_fits: bool = False,
    min_snr: float = 3.5,
    attempt_reverse_nergative_flood_fill: bool = True,
    overwrite: bool = True,
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
        min_snr (float, optional): Minimum signal-to-noise ratio for the masking to include a pixel. Defaults to 3.5.
        attempt_negative_flood_fill (bool): Attempt to filter out negative sidelobes from the bask. See `reverse_negative_flood_fill`. Defaults to True.
        overwrite (bool): Passed to `fits.writeto`, and will overwrite files should they exist. Defaults to True.

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
            filename=mask_names.signal_fits,
            data=signal_data,
            header=fits_header,
            overwrite=overwrite,
        )

    # Following the help in wsclean:
    # WSClean accepts masks in CASA format and in fits file format. A mask is a
    # normal, single polarization image file, where all zero values are interpreted
    # as being not masked, and all non-zero values are interpreted as masked. In the
    # case of a fits file, the file may either contain a single frequency or it may
    # contain a cube of images.
    if attempt_reverse_nergative_flood_fill:
        mask_data = reverse_negative_flood_fill(
            signal=np.squeeze(signal_data),
            positive_seed_clip=4,
            positive_flood_clip=1.5,
            negative_seed_clip=4,
            guard_negative_dilation=30,
            grow_low_snr=1.5,
            grow_low_island_size=768,
        )
        mask_data = mask_data.reshape(signal_data.shape)
    else:
        logger.info(f"Clipping using a {min_snr=}")
        mask_data = (signal_data > min_snr).astype(np.int32)

    logger.info(f"Writing {mask_names.mask_fits}")
    fits.writeto(
        filename=mask_names.mask_fits,
        data=mask_data,
        header=fits_header,
        overwrite=overwrite,
    )

    return mask_names


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Simple utility functions to create masks from FITS images. "
    )

    subparser = parser.add_subparsers(
        dest="mode", help="Operation mode of flint_bandpass"
    )

    fits_parser = subparser.add_parser(
        "snrmask",
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
    fits_parser.add_argument(
        "--use-butterworth",
        action="store_true",
        help="Apply a butterworth filter to smooth the total intensity image before computing the signal map. ",
    )
    fits_parser.add_argument(
        "--connectivity-shape",
        default=(4, 4),
        nargs=2,
        type=int,
        help="The connectivity matrix to use when applying a binary erosion. Only used when using the butterworth smoothing filter. ",
    )

    extract_parser = subparser.add_parser(
        "extractmask",
        help="Extract a beam FITS masked region from a larger FITS mask mosaic image. ",
    )

    extract_parser.add_argument(
        "beam_image",
        type=Path,
        help="The FITS image with the WCS that will be used to extract the mask region from",
    )

    extract_parser.add_argument(
        "mosaic_image",
        type=Path,
        help="The field mosaic image that will be used as a basis to extract a region mask from",
    )

    return parser


def cli():
    parser = get_parser()

    args = parser.parse_args()

    if args.mode == "snrmask":
        if args.user_butterworth:
            create_snr_mask_wbutter_from_fits(
                fits_image_path=args.image,
                fits_rms_path=args.rms,
                fits_bkg_path=args.bkg,
                create_signal_fits=args.save_signal,
                min_snr=args.min_snr,
                connectivity_shape=tuple(args.connectivity_shape),
            )
        else:
            create_snr_mask_from_fits(
                fits_image_path=args.image,
                fits_rms_path=args.rms,
                fits_bkg_path=args.bkg,
                create_signal_fits=args.save_signal,
                min_snr=args.min_snr,
            )

    elif args.mode == "extractmask":
        extract_beam_mask_from_mosaic(
            fits_beam_image_path=args.beam_image,
            fits_mosaic_mask_names=FITSMaskNames(mask_fits=args.mosaic_image),
        )
    else:
        logger.error(f"Supplied mode {args.mode} is not known. ")


if __name__ == "__main__":
    cli()
