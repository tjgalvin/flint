"""Utility functions to make image based masks from images, with the initial
thought being towards FITS images.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Collection, NamedTuple, Optional, Union

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
from scipy.ndimage import label, minimum_filter

from flint.logging import logger
from flint.naming import FITSMaskNames, create_fits_mask_names
from flint.utils import get_pixels_per_beam


class MaskingOptions(NamedTuple):
    """Contains options for the creation of clean masks from some subject
    image. Clipping levels specified are in units of RMS (or sigma). They
    are NOT in absolute units.
    """

    base_snr_clip: float = 4
    """A base clipping level to be used should other options not be activated"""
    flood_fill: bool = True
    """Whether to attempt to flood fill when constructing a mask. This should be `True` for `grow_low_snr_islands` and `suppress_artefacts   to have an effect. """
    flood_fill_positive_seed_clip: float = 4.5
    """The clipping level to seed islands that will be grown to lower SNR"""
    flood_fill_positive_flood_clip: float = 1.5
    """Clipping level used to grow seeded islands down to"""
    flood_fill_use_mbc: bool = False
    """If True, the clipping levels are used as the `increase_factor` when using a minimum absolute clip"""
    flood_fill_use_mbc_box_size: int = 75
    """The size of the mbc box size should mbc be used"""
    suppress_artefacts: bool = True
    """Whether to attempt artefacts based on the presence of significant negatives"""
    suppress_artefacts_negative_seed_clip: float = 5
    """The significance level of a negative island for the sidelobe suppression to be activated. This should be a positive number (the signal map is internally inverted)"""
    suppress_artefacts_guard_negative_dilation: float = 40
    """The minimum positive significance pixels should have to be guarded when attempting to suppress artefacts around bright sources"""
    suppress_artefacts_large_island_threshold: float = 1.0
    """Threshold in units of beams for an island of negative pixels to be considered large"""
    grow_low_snr_island: bool = False
    """Whether to attempt to grow a mask to capture islands of low SNR (e.g. diffuse emission)"""
    grow_low_snr_island_clip: float = 1.75
    """The minimum significance levels of pixels to be to seed low SNR islands for consideration"""
    grow_low_snr_island_size: int = 768
    """The number of pixels an island has to be for it to be accepted"""
    minimum_boxcar: bool = True
    """Use the boxcar minimum threshold to compare to remove artefacts"""
    minimum_boxcar_size: int = 100
    """Size of the boxcar filter"""
    minimum_boxcar_increase_factor: float = 1.2
    """The factor used to construct minimum positive signal threshold for an island """

    def with_options(self, **kwargs) -> MaskingOptions:
        """Return a new instance of the MaskingOptions"""
        _dict = self._asdict()
        _dict.update(**kwargs)

        return MaskingOptions(**_dict)


def consider_beam_mask_round(
    current_round: int,
    mask_rounds: Union[str, Collection[int], int],
    allow_beam_masks: bool = True,
) -> bool:
    """Evaluate whether a self-calibration round should have a beam clean mask
    constructed. Rules are:

    - if `mask_rounds` is a string and is "all", all rounds will have a beam mask
    - if 'mask_rounds' is a single integer, so long as `current_round` is larger it will have a beam mask
    - if `mask_rounds` is iterable and contains `current_round` it will have a beam mask
    - if `allow_beam_masks` is False a False is returned. Otherwise options above are considered.

    Args:
        current_round (int): The current self-calibration round that is being performed
        mask_rounds (Union[str, Collection[int], int]): The rules to consider whether a beam mask is needed
        allow_beam_masks (bool, optional): A global allow / deny. This should be `True` for other rules to be considered. Defaults to True.

    Returns:
        bool: Whether per beam mask should be performed
    """
    logger.info(f"Considering {current_round=} {mask_rounds=} {allow_beam_masks=}")

    # The return below was getting silly to meantally parse
    if not allow_beam_masks:
        return False

    return mask_rounds is not None and (
        (isinstance(mask_rounds, str) and mask_rounds.lower() == "all")
        or (isinstance(mask_rounds, int) and current_round >= mask_rounds)
        or (isinstance(mask_rounds, (list, tuple)))
        and current_round in mask_rounds
    )  # type: ignore


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
    # TODO: Ideally we can accept an arbitrary WCS, or read the wsclean docs to
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

    if signal is None and image is not None and rms is not None:
        if background is None:
            logger.info("No background supplied, assuming zeros. ")
            background = np.zeros_like(image)

        out_signal = (image - background) / rms
    else:
        out_signal = signal

    return out_signal


def grow_low_snr_mask(
    image: Optional[np.ndarray] = None,
    rms: Optional[np.ndarray] = None,
    background: Optional[np.ndarray] = None,
    signal: Optional[np.ndarray] = None,
    grow_low_snr: float = 2.0,
    grow_low_island_size: int = 512,
    region_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """There may be cases where normal thresholding operations based on simple pixel-wise SNR
    cuts fail to pick up diffuse, low surface brightness regions of emission. When some type
    of masking operation is used there may be instances where these regions are never cleaned.

    Sometimes smoothing can help to pick up these features, but when attempting to pass such
    a mask through to the imagery of choice the islands may be larger than the features at their
    native resolution, unless some other more sophisticated filtering is performed.

    This function attempts to grow masks to capture islands of contiguous pixels above a low
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
        np.ndarray: The derived mask of objects with low-surface brightness
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

    small_islands = [
        idx
        for idx, count in enumerate(counts)
        if count < grow_low_island_size and idx > 0
    ]
    low_snr_mask[np.isin(mask_labels, small_islands)] = False

    return low_snr_mask


# TODO> Allow box car size to be scaled in proportion to beam size
def minimum_boxcar_artefact_mask(
    signal: np.ndarray,
    island_mask: np.ndarray,
    boxcar_size: int,
    increase_factor: float = 2.0,
) -> np.ndarray:
    """Attempt to remove islands from a potential clean mask by
    examining surrounding pixels. A boxcar is applied to find the
    minimum signal to noise in a small localised region. For each
    island the maximum signal is considered.

    If the absolute minimum signal increased by a factor in an
    island region is larger to the maximum signal then that island
    is omitted.


    Args:
        signal (np.ndarray): The input signl to use
        island_mask (np.ndarray): The current island mask derived by other methods
        boxcar_size (int): Size of the minimum boxcar size
        increase_factor (float, optional): Factor to increase the minimum signal by. Defaults to 2.0.

    Returns:
        np.ndarray: _description_
    """

    logger.info(
        f"Running boxcar minimum island clip with {boxcar_size=} {increase_factor=}"
    )

    # Make a copy of the original island mask to avoid unintended persistence
    mask = island_mask.copy()

    # label each of the islands with a id
    mask_labels, _ = label(island_mask, structure=np.ones((3, 3)))  # type: ignore
    uniq_labels = np.unique(mask_labels)
    logger.info(f"Number of unique islands: {len(uniq_labels)}")

    rolling_min = minimum_filter(signal, boxcar_size)

    # For each island work out the maximum signal in the island and the minimum signal
    # at the island in the output of the boxcar.
    island_min, island_max = {}, {}
    for island_id in uniq_labels:
        if island_id == 0:
            continue

        # compute the mask once. These could be a dixt comprehension
        # but then this mask is computed twice
        island_id_mask = mask_labels == island_id

        island_max[island_id] = np.max(signal[island_id_mask])
        island_min[island_id] = np.min(rolling_min[island_id_mask])

    # Nuke the weak ones, mask and report
    eliminate = [
        k
        for k in island_max
        if island_max[k] < np.abs(increase_factor * island_min[k])
        and island_min[k] < 0.0
    ]
    # Walk the plank
    mask[np.isin(mask_labels, eliminate)] = False

    logger.info(f"Eliminated {len(eliminate)} islands")

    return mask


def suppress_artefact_mask(
    signal: np.ndarray,
    negative_seed_clip: float,
    guard_negative_dilation: float,
    pixels_per_beam: Optional[float] = None,
    large_island_threshold: float = 1.0,
) -> np.ndarray:
    """Attempt to grow mask that sepresses artefacts around bright sources. Small islands
    of negative emission seed pixels, which then grow out. Bright positive pixels are not
    allowed to change (which presumably are the source of negative artetfacts).

    The assumption here is that:

    - no genuine source of negative sky emission
    - negative islands are around bright sources with deconvolution/calibration errors
    - if there are brightish negative islands there is also positive brightish arteefact islands nearby

    For this reason the guard mask should be sufficiently high to protect the main source but nuke the fask positive islands


    Args:
        signal (np.ndarray): The signal mask,
        negative_seed_clip (float): The minimum significance level to seed. This is a positive number (as it is applied to the inverted signal).
        guard_negative_dilation (float): Regions of positive emission above this are protected. This is positive.
        pixels_per_beam (Optional[float], optional): The number of pixels per beam. If not None, seed islands larger than this many pixels are removed. Defaults to None.
        large_island_threshold (float, optional): The number of beams required for a large island of negative pixels to be dropped as an artefact seed. Only used if `pixels_per_beam` is set. Defaults to 1.0.

    Returns:
        np.ndarray: The artefact suppression mask
    """
    # This Pirate thinks provided the background is handled
    # that taking the inverse is correct
    negative_signal = -1 * signal

    negative_mask = negative_signal > negative_seed_clip

    if pixels_per_beam:
        mask_labels, no_labels = label(negative_mask, structure=np.ones((3, 3)))
        _, counts = np.unique(mask_labels.flatten(), return_counts=True)

        clip_pixels_threshold = large_island_threshold * pixels_per_beam
        logger.info(
            f"Removing negative islands larger than {clip_pixels_threshold} pixels with {large_island_threshold=}, {pixels_per_beam=}"
        )

        large_islands = [
            idx
            for idx, count in enumerate(counts)
            if count > clip_pixels_threshold and idx > 0
        ]
        logger.info(f"Removing islands with labels: {large_islands=}")
        negative_mask[np.isin(mask_labels, large_islands)] = False

    negative_dilated_mask = scipy_binary_dilation(
        input=negative_mask,
        mask=signal < guard_negative_dilation,
        iterations=10,
        structure=np.ones((3, 3)),
    )

    return negative_dilated_mask


def minimum_absolute_clip(
    image: np.ndarray, increase_factor: float = 2.0, box_size: int = 100
) -> np.ndarray:
    """Given an input image or signal array, construct a simple image mask by applying a
    rolling boxcar minimum filter, and then selecting pixels above a cut of
    the absolute value value scaled by `increase_factor`. This is a pixel-wise operation.

    Args:
        image (np.ndarray): The input array to consider
        increase_factor (float, optional): How large to scale the absolute minimum by. Defaults to 2.0.
        box_size (int, optional): Size of the rolling boxcar minimum filtr. Defaults to 100.

    Returns:
        np.ndarray: The mask of pixels above the locally varying threshold
    """
    logger.info(f"Minimum absolute clip, {increase_factor=} {box_size=}")
    rolling_box_min = minimum_filter(image, box_size)

    image_mask = image > (increase_factor * np.abs(rolling_box_min))
    # NOTE: This used to attempt to select pixels should that belong to an island of positive pixels with a box that was too small
    # | (
    #     (image > 0.0) & (rolling_box_min > 0.0)
    # )

    return image_mask


def _verify_set_positive_seed_clip(
    positive_seed_clip: float, signal: np.ndarray
) -> float:
    """Ensure that the positive seed clip is handled appropriately"""
    max_signal = np.max(signal)
    if max_signal < positive_seed_clip:
        logger.critical(
            (
                f"The maximum signal {max_signal:.4f} is below the provided {positive_seed_clip=}. "
                "Setting clip to 90 percent of maximum. "
            )
        )
        positive_seed_clip = max_signal * 0.9

    return positive_seed_clip


def reverse_negative_flood_fill(
    masking_options: MaskingOptions,
    image: Optional[np.ndarray] = None,
    rms: Optional[np.ndarray] = None,
    background: Optional[np.ndarray] = None,
    signal: Optional[np.ndarray] = None,
    pixels_per_beam: Optional[float] = None,
) -> np.ndarray:
    """Attempt to:

    * seed masks around bright regions of an image and grow them to lower significance thresholds
    * remove regions of negative and positive islands that surround bright sources.

    An initial set of islands (and masks) are constructed by first
    using the `positive_seed_clip` to create an initial SNR based
    mask. These islands then are binary dilated to grow the islands
    to adjacent pixels at a lower significance level (see `scipy.ndimage.binary_dilation`).

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
        masking_options (MaskingOptions): Options to carry out masking.
        image (Optional[np.ndarray], optional): The total intensity pixels to have the mask for. Defaults to None.
        rms (Optional[np.ndarray], optional): The noise across the image. Defaults to None.
        background (Optional[np.ndarray], optional): The background acros the image. If None, zeros are assumed. Defaults to None.
        signal(Optional[np.ndarray], optional): A signal map. Defaults to None.
        pixels_per_beam (Optional[float], optional): The number of pixels that cover a beam. Defaults to None.

    Returns:
        np.ndarray: Mask of the pixels to clean
    """

    logger.info("Will be reversing flood filling")
    logger.info(f"{masking_options=} ")

    signal = _get_signal_image(
        image=image, rms=rms, background=background, signal=signal
    )

    if masking_options.flood_fill_use_mbc and image is not None:
        positive_mask = minimum_absolute_clip(
            image=image,
            increase_factor=masking_options.flood_fill_positive_seed_clip,
            box_size=masking_options.flood_fill_use_mbc_box_size,
        )
        flood_floor_mask = minimum_absolute_clip(
            image=image,
            increase_factor=masking_options.flood_fill_positive_flood_clip,
            box_size=masking_options.flood_fill_use_mbc_box_size,
        )
    else:
        # Sanity check the upper clip level, you rotten seadog
        positive_seed_clip = _verify_set_positive_seed_clip(
            positive_seed_clip=masking_options.flood_fill_positive_seed_clip,
            signal=signal,
        )
        # Here we create the mask image that will start the binary dilation
        # process, and we will ensure only pixels above the `positive_flood_clip`
        # are allowed to be dilated. In other words we are growing the mask
        positive_mask = signal >= positive_seed_clip
        flood_floor_mask = signal > masking_options.flood_fill_positive_flood_clip

    positive_dilated_mask = scipy_binary_dilation(
        input=positive_mask,
        mask=flood_floor_mask,
        iterations=1000,
        structure=np.ones((3, 3)),
    )

    if masking_options.minimum_boxcar:
        positive_dilated_mask = minimum_boxcar_artefact_mask(
            signal=signal,
            island_mask=positive_dilated_mask,
            boxcar_size=masking_options.minimum_boxcar_size,
            increase_factor=masking_options.minimum_boxcar_increase_factor,
        )

    negative_dilated_mask = None
    if masking_options.suppress_artefacts:
        negative_dilated_mask = suppress_artefact_mask(
            signal=signal,
            negative_seed_clip=masking_options.suppress_artefacts_negative_seed_clip,
            guard_negative_dilation=masking_options.suppress_artefacts_guard_negative_dilation,
            pixels_per_beam=pixels_per_beam,
            large_island_threshold=masking_options.suppress_artefacts_large_island_threshold,
        )

        # and here we set the presumable nasty islands to False
        positive_dilated_mask[negative_dilated_mask] = False

    if masking_options.grow_low_snr_island:
        low_snr_mask = grow_low_snr_mask(
            signal=signal,
            grow_low_snr=masking_options.grow_low_snr_island_clip,
            grow_low_island_size=masking_options.grow_low_snr_island_size,
            region_mask=negative_dilated_mask,
        )
        positive_dilated_mask[low_snr_mask] = True

    return positive_dilated_mask.astype(np.int32)


def create_snr_mask_from_fits(
    fits_image_path: Path,
    fits_rms_path: Path,
    fits_bkg_path: Path,
    masking_options: MaskingOptions,
    create_signal_fits: bool = False,
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
        masking_options (MaskingOptions): Configurables on the masking operation procedure.
        create_signal_fits (bool, optional): Create an output signal map. Defaults to False.
        overwrite (bool): Passed to `fits.writeto`, and will overwrite files should they exist. Defaults to True.

    Returns:
        FITSMaskNames: Container describing the signal and mask FITS image paths. If ``create_signal_path`` is None, then the ``signal_fits`` attribute will be None.
    """
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

    pixels_per_beam = get_pixels_per_beam(fits_path=fits_image_path)

    # Following the help in wsclean:
    # WSClean accepts masks in CASA format and in fits file format. A mask is a
    # normal, single polarization image file, where all zero values are interpreted
    # as being not masked, and all non-zero values are interpreted as masked. In the
    # case of a fits file, the file may either contain a single frequency or it may
    # contain a cube of images.
    if masking_options.flood_fill:
        # TODO: This function should really just accept a MaskingOptions directly
        mask_data = reverse_negative_flood_fill(
            signal=np.squeeze(signal_data),
            image=np.squeeze(fits.getdata(fits_image_path)),
            masking_options=masking_options,
            pixels_per_beam=pixels_per_beam,
        )
        mask_data = mask_data.reshape(signal_data.shape)
    else:
        logger.info(f"Clipping using a {masking_options.base_snr_clip=}")
        mask_data = (signal_data > masking_options.base_snr_clip).astype(np.int32)

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
        masking_options = MaskingOptions(base_snr_clip=args.min_snr)
        create_snr_mask_from_fits(
            fits_image_path=args.image,
            fits_rms_path=args.rms,
            fits_bkg_path=args.bkg,
            create_signal_fits=args.save_signal,
            masking_options=masking_options,
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
