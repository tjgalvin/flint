"""This is an interface into the yandasoft linmos task."""

from argparse import ArgumentParser
from pathlib import Path
from typing import Collection, List, NamedTuple, Optional, Tuple

import numpy as np
from astropy.io import fits

from flint.logging import logger
from flint.naming import LinmosNames, create_linmos_names, extract_beam_from_name
from flint.sclient import run_singularity_command

# This is the expected orientation of the third-axis and footprint (remember the footprint
# can be electronically rotated).
EXPECTED_HOLOGRAPHY_ROTATION_CONSTANT_RADIANS = -2 * np.pi / 8


class LinmosCommand(NamedTuple):
    cmd: str
    """The yandasoft linmos task that will be executed"""
    parset: Path
    """The output location that the generated linmos parset has been written to"""
    image_fits: Path
    """Path to the output linmos image created (or will be). """


class BoundingBox(NamedTuple):
    """Simple container to represent a bounding box"""

    xmin: int
    """Minimum x pixel"""
    xmax: int
    """Maximum x pixel"""
    ymin: int
    """Minimum y pixel"""
    ymax: int
    """Maximum y pixel"""
    original_shape: Tuple[int, int]
    """The original shape of the image"""


def create_bound_box(image_data: np.ndarray, is_masked: bool = False) -> BoundingBox:
    """Construct a bounding box around finite pixels.

    If ``is_mask` is ``False``, the ``image_data`` will be masked internally using ``numpy.isfinite``.

    Args:
        image_data (np.ndarray): The image data that will have a bounding box constructed for.
        is_masked (bool, optional): if this is ``True`` the ``image_data`` are treated as a boolean mask array. Defaults to False.

    Returns:
        BoundingBox: The tight bounding box around pixels.
    """
    assert (
        len(image_data.shape) == 2
    ), f"Only two-dimensional arrays supported, received {image_data.shape}"

    # First convert to a boolean array
    image_valid = image_data if is_masked else np.isfinite(image_data)

    # Then make them 1D arrays
    x_valid = np.any(image_valid, axis=1)
    y_valid = np.any(image_valid, axis=0)

    # Now get the first and last index
    xmin, xmax = np.where(x_valid)[0][[0, -1]]
    ymin, ymax = np.where(y_valid)[0][[0, -1]]

    return BoundingBox(
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, original_shape=image_data.shape
    )


class TrimImageResult(NamedTuple):
    """The constructed path and the bounding box"""

    path: Path
    """The path to the trimmed image"""
    bounding_box: BoundingBox
    """The bounding box that was applied to the image"""


def trim_fits_image(
    image_path: Path, bounding_box: Optional[BoundingBox] = None
) -> TrimImageResult:
    """Trim the FITS image produces by linmos to remove as many empty pixels around
    the border of the image as possible. This is an inplace operation.

    Args:
        image_path (Path): The FITS image that will have its border trimmed
        bounding_box (Optional[BoundingBox], optional): The bounding box that will be applied to the image. If None it is computed. Defaults to None.

    Returns:
        Path: Path of the FITS image that had its border trimmed
    """
    logger.info(f"Trimming {image_path.name}")
    with fits.open(image_path) as fits_image:
        data = fits_image[0].data
        logger.info(f"Original data shape: {data.shape}")

        image_shape = (data.shape[-2], data.shape[-1])
        logger.info(f"The image dimensions are: {image_shape}")

        if not bounding_box:
            bounding_box = create_bound_box(
                image_data=np.squeeze(data), is_masked=False
            )
        else:
            if image_shape != bounding_box.original_shape:
                raise ValueError(
                    f"Bounding box constructed against {bounding_box.original_shape}, but being applied to {image_shape=}"
                )

        data = data[
            ...,
            bounding_box.xmin : bounding_box.xmax,
            bounding_box.ymin : bounding_box.ymax,
        ]

        header = fits_image[0].header
        header["CRPIX1"] -= bounding_box.ymin
        header["CRPIX2"] -= bounding_box.xmin

        logger.info(f"Trimmed data shape: {data.shape}")

    fits.writeto(filename=image_path, data=data, header=header, overwrite=True)

    return TrimImageResult(path=image_path, bounding_box=bounding_box)


def get_image_weight(
    image_path: Path, mode: str = "mad", image_slice: int = 0
) -> float:
    """Compute an image weight supplied to linmos, which is used for optimally
    weighting overlapping images. Supported modes are 'mad' and 'mtd', which
    simply resolve to their numpy equivalents.

    This weight is really a relative weight to used between all images in a set
    of images being co-added together. So long as these are all calculated in
    the same way, it does not necessarily have to correspond to an optimatelly
    calculated RMS.

    Args:
        image (Path): The path to the image fits file to inspect.
        mode (str, optional): Which mode should be used when calculating the weight. Defaults to 'mad'.
        image_slice (int, optional): The image slice in the HDU list of the `image` fits file to inspect. Defaults to 0.

    Raises:
        ValueError: Raised when a mode is requested but does not exist

    Returns:
        float: The weight to supply to linmos
    """

    logger.debug(
        f"Computing linmos weight using {mode=}, {image_slice=} for {image_path}. "
    )
    weight_modes = ("mad", "std")
    assert (
        mode in weight_modes
    ), f"Invalid {mode=} specified. Available modes: {weight_modes}"

    with fits.open(image_path, memmap=True) as in_fits:
        image_data = in_fits[image_slice].data

        assert len(
            image_data.shape
        ), f"{len(image_data.shape)=} is less than two. Is this really an image?"

        # remove non-finite numbers
        image_data = image_data[np.isfinite(image_data)]

        logger.debug(f"Data shape is: {image_data.shape}")
        if mode == "mad":
            median = np.median(image_data)
            mad = np.median(np.abs(image_data - median))
            weight = 1.0 / mad**2
        elif mode == "std":
            std = np.std(image_data)
            weight = 1.0 / std**2
        else:
            raise ValueError(
                f"Invalid {mode=} specified. Available modes: {weight_modes}"
            )

    logger.info(f"Weight {weight:.3f} for {image_path}")
    return weight


def generate_weights_list_and_files(
    image_paths: Collection[Path], mode: str = "mad"
) -> str:
    """Generate the expected linmos weight files, and construct an appropriate
    string that can be embedded into a linmos partset. These weights files will
    appear as:

    >>> #Channel Weight
    >>> 0 1234.5
    >>> 1 6789.0

    The weights should be correct relative to the entire set of input images.
    They do not necessarily have to correspond to an accurate measure of the RMS.

    This function will create a corresponding text file for each input image. At
    the moment it is only intended to work on MFS images. It __is not__ currently
    intended to be used on image cubes.

    Args:
        image_paths (Collection[Path]): Images to iterate over to create a corresponding weights.txt file.
        mode (str, optional): The mode to use when calling get_image_weight

    Returns:
        str: The string to embedded into the yandasoft linmos parset file
    """
    logger.info(
        f"No weights provided. Calculating weights for {len(image_paths)} images."
    )

    # TODO: image cubes should be supported here. This would required iterating
    # over each channel in the FITS cube.
    weight_file_list = []
    for image in image_paths:
        weight_file = image.with_suffix(".weights.txt")
        weight_file_list.append(weight_file)

        # Must be of the format:
        # #Channel Weight
        # 0 1234.5
        # 1 6789.0
        with open(weight_file, "w") as out_file:
            logger.info(f"Writing {weight_file}")
            out_file.write("#Channel Weight\n")
            image_weight = get_image_weight(image_path=image, mode=mode)
            out_file.write(f"0 {image_weight}\n")

    weight_str = [
        str(weight_file) for weight_file in weight_file_list if weight_file.exists()
    ]
    weight_list = "[" + ",".join(weight_str) + "]"

    return weight_list


def generate_linmos_parameter_set(
    images: Collection[Path],
    parset_output_path: Path,
    linmos_names: LinmosNames,
    weight_list: Optional[str] = None,
    holofile: Optional[Path] = None,
    cutoff: float = 0.001,
    pol_axis: Optional[float] = None,
    overwrite: bool = True,
) -> Path:
    """Generate a parset file that will be used with the
    yandasoft linmos task.

    Args:
        images (Collection[Path]): The images that will be coadded into a single field image.
        parset_output_path (Path): Path of the output linmos parset file.
        linmos_names (LinmosNames): Names of the output image and weights that linmos will produces. The weight image will have a similar name. Defaults to "linmos_field".
        weight_list (str, optional): If not None, this string will be embedded into the yandasoft linmos parset as-is. It should represent the formatted string pointing to weight files, and should be equal length of the input images. If None it is internally generated. Defaults to None.
        holofile (Optional[Path], optional): Path to a FITS cube produced by the holography processing pipeline. Used by linmos to appropriate primary-beam correct the images. Defaults to None.
        cutoff (float, optional): Pixels whose primary beam attenuation is below this cutoff value are blanked. Defaults to 0.001.
        pol_axis (Optional[float], optional): The physical orientation of the ASKAP third-axis. This is provided (with some assumptions about the orientation of the holography) to correctly rotate the attenuation of the beams when coadding. If None we hope for the best. Defaults to None.
        overwrite (bool, optional): If True and the parset file already exists, overwrite it. Otherwise a FileExistsError is raised should the parset exist. Defaults to True.

    Returns:
        Path: Path to the output parset file.
    """

    # TODO: Add the alpha linmos param

    img_str: List[str] = list(
        [str(p).replace(".fits", "") for p in images if p.exists()]
    )
    logger.info(f"{len(img_str)} unique images from {len(images)} input collection. ")
    img_list: str = "[" + ",".join(img_str) + "]"

    assert (
        len(set(img_str)) == len(images)
    ), "Some images were dropped from the linmos image string. Something is bad, walk the plank. "

    # If no weights_list has been provided (and therefore no optimal
    # beam-wise weighting) assume that all beams are of about the same
    # quality. In reality, this should be updated to provide a RMS noise
    # estimate per-pixel of each image.
    if weight_list is None:
        weight_list = generate_weights_list_and_files(image_paths=images, mode="mad")

    beam_order_strs = [str(extract_beam_from_name(str(p.name))) for p in images]
    beam_order_list = "[" + ",".join(beam_order_strs) + "]"

    # The yandasoft linmos tasks will insist on adding a .fits extension
    # so it needs to be dropped from the Path objects. Using the Path.stem
    # attribute drops the parent directory (absolute directory).
    parent_dir = linmos_names.image_fits.parent

    # Work out the appropriate rotation to provide linmos. This should
    # be the differential pol. axis. rotation between the science field
    # and the holography. At the moment this holography rotation is
    # unknown to us (not stored in the cube header).
    alpha = None
    if pol_axis:
        logger.info(
            f"The constant assumed holography rotation is: {EXPECTED_HOLOGRAPHY_ROTATION_CONSTANT_RADIANS:.4f} radians"
        )
        logger.info(f"The extracted pol_axis of the field: {pol_axis:.4f} radians")
        alpha = pol_axis - EXPECTED_HOLOGRAPHY_ROTATION_CONSTANT_RADIANS
        logger.info(f"Differential rotation is: {alpha} rad")

    # Parameters are taken from arrakis
    parset = (
        f"linmos.names            = {img_list}\n"
        f"linmos.weights          = {weight_list}\n"
        f"linmos.beams            = {beam_order_list}\n"
        # f"linmos.beamangle        = {beam_angle_list}\n"
        f"linmos.imagetype        = fits\n"
        f"linmos.outname          = {str(parent_dir / linmos_names.image_fits.stem)}\n"
        f"linmos.outweigh        = {str(parent_dir / linmos_names.weight_fits.stem)}\n"
        f"# For ASKAPsoft>1.3.0\n"
        f"linmos.useweightslog    = true\n"
        f"linmos.weighttype       = Combined\n"
        f"linmos.weightstate      = Inherent\n"
        f"linmos.cutoff           = {cutoff}\n"
    )

    # This requires an appropriate holography fits cube to
    # have been created. This is typically done outside as
    # part of operations.
    if holofile is not None:
        logger.info(f"Using holography file {holofile} -- setting removeleakge to true")

        assert holofile.exists(), f"{holofile=} has been specified but does not exist. "

        parset += (
            f"linmos.primarybeam      = ASKAP_PB\n"
            f"linmos.primarybeam.ASKAP_PB.image = {str(holofile.absolute())}\n"
            # f"linmos.primarybeam.ASKAP_PB.alpha = {paf_alpha}\n"
            f"linmos.removeleakage    = true\n"
        )
        if alpha:
            parset += f"linmos.primarybeam.ASKAP_PB.alpha = {alpha} # in radians\n"

    # Now write the file, me hearty
    logger.info(f"Writing parset to {str(parset_output_path)}.")
    logger.info(f"{parset}")
    if not overwrite:
        assert not Path(
            parset_output_path
        ).exists(), f"The parset {parset_output_path} already exists!"
    with open(parset_output_path, "w") as parset_file:
        parset_file.write(parset)

    return parset_output_path


def linmos_images(
    images: Collection[Path],
    parset_output_path: Path,
    image_output_name: str = "linmos_field",
    weight_list: Optional[str] = None,
    holofile: Optional[Path] = None,
    container: Path = Path("yandasoft.sif"),
    cutoff: float = 0.001,
    pol_axis: Optional[float] = None,
) -> LinmosCommand:
    """Create a linmos parset file and execute it.

    Args:
        images (Collection[Path]): The images that will be coadded into a single field image.
        parset_output_path (Path): Path of the output linmos parset file.
        image_output_name (str, optional): Name of the output image linmos produces. The weight image will have a similar name. Defaults to "linmos_field".
        weight_list (str, optional): If not None, this string will be embedded into the yandasoft linmos parset as-is. It should represent the formatted string pointing to weight files, and should be equal length of the input images. If None it is internally generated. Defaults to None.
        holofile (Optional[Path], optional): Path to a FITS cube produced by the holography processing pipeline. Used by linmos to appropriate primary-beam correct the images. Defaults to None.
        container (Path, optional): Path to the singularity container that has the yandasoft tools. Defaults to Path('yandasoft.sif').
        cutoff (float, optional): Pixels whose primary beam attenuation is below this cutoff value are blanked. Defaults to 0.001.
        pol_axis (Optional[float], optional): The physical oritentation of the ASKAP third-axis in radians. Defaults to None.

    Returns:
        LinmosCommand: The linmos command executed and the associated parset file
    """

    assert (
        container.exists()
    ), f"The yandasoft container {str(container)} was not found. "

    linmos_names: LinmosNames = create_linmos_names(name_prefix=image_output_name)

    linmos_parset = generate_linmos_parameter_set(
        images=images,
        parset_output_path=parset_output_path,
        linmos_names=linmos_names,
        weight_list=weight_list,
        holofile=holofile,
        cutoff=cutoff,
        pol_axis=pol_axis,
    )

    linmos_cmd_str = f"linmos -c {str(linmos_parset)}"
    bind_dirs = [image.absolute() for image in images] + [linmos_parset.absolute()]
    if holofile:
        bind_dirs.append(holofile.absolute())

    run_singularity_command(
        image=container, command=linmos_cmd_str, bind_dirs=bind_dirs
    )

    linmos_cmd = LinmosCommand(
        cmd=linmos_cmd_str,
        parset=linmos_parset,
        image_fits=linmos_names.image_fits.absolute(),
    )

    # Trim the fits image to remove empty pixels
    image_trim_results = trim_fits_image(image_path=linmos_names.image_fits)
    trim_fits_image(
        image_path=linmos_names.weight_fits,
        bounding_box=image_trim_results.bounding_box,
    )

    return linmos_cmd


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)

    subparsers = parser.add_subparsers(dest="mode")

    parset_parser = subparsers.add_parser(
        "parset", help="Generate a yandasoft linmos parset"
    )

    parset_parser.add_argument(
        "images", type=Path, nargs="+", help="The images that will be coadded"
    )
    parset_parser.add_argument(
        "parset_output_path", type=Path, help="The output path of the linmos parser"
    )
    parset_parser.add_argument(
        "--image-output-name",
        type=str,
        default="linmos_field",
        help="The base name used to create the output linmos images and weight maps",
    )
    parset_parser.add_argument(
        "--weight-list",
        type=Path,
        default=None,
        help="Path a new-line delimited text-file containing the relative weights corresponding to the input images",
    )
    parset_parser.add_argument(
        "--holofile",
        type=Path,
        default=None,
        help="Path to the holography FITS cube used for primary beam corrections",
    )
    parset_parser.add_argument(
        "--yandasoft-container",
        type=Path,
        default=None,
        help="Path to the container with yandasoft tools",
    )

    trim_parser = subparsers.add_parser(
        "trim", help="Generate a yandasoft linmos parset"
    )

    trim_parser.add_argument(
        "images", type=Path, nargs="+", help="The images that will be trimmed"
    )
    return parser


def cli() -> None:
    parser = get_parser()

    args = parser.parse_args()

    if args.mode == "parset":
        if args.yandasoft_container is None:
            linmos_names = create_linmos_names(name_prefix=args.image_output_name)
            generate_linmos_parameter_set(
                images=args.images,
                parset_output_path=args.parset_output_path,
                linmos_names=linmos_names,
                weight_list=args.weight_list,
                holofile=args.holofile,
            )
        else:
            linmos_images(
                images=args.images,
                parset_output_path=args.parset_output_path,
                image_output_name=args.image_output_name,
                weight_list=args.weight_list,
                holofile=args.holofile,
                container=args.yandasoft_container,
            )
    elif args.mode == "trim":
        images = args.images
        logger.info(f"Will be trimming {len(images)}")
        for image in images:
            trim_fits_image(image_path=Path(image))
    else:
        logger.error(f"Unrecognised mode: {args.mode}")


if __name__ == "__main__":
    cli()
