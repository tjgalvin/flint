"""This is an interface into the yandasoft linmos task. 
"""
from pathlib import Path
from typing import Optional, Collection, List, NamedTuple
from argparse import ArgumentParser

import numpy as np
from astropy.io import fits

from flint.logging import logger
from flint.sclient import run_singularity_command
from flint.naming import extract_beam_from_name

class LinmosCMD(NamedTuple):
    cmd: str
    """The yandasoft linmos task that will be executed"""
    parset: Path
    """The output location that the generated linmos parset has been writen to"""

def get_image_weight(image_path: Path, mode: str='mad', image_slice: int=0) -> float:
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

    logger.info(f"Compuuting linmos weight using {mode=}, {image_slice=} for {image_path}. ")
    weight_modes = ('mad', 'std')

    with fits.open(image_path, memmap=True) as in_fits:
        image_data = in_fits[image_slice].data
        
        assert len(image_data.shape), f"{len(image_data.shape)=} is less than two. Is this really an image?"
        
        logger.info(f"Data shape is: {image_data.shape}")
        if mode == 'mad':
            median = np.median(image_data)
            weight = np.median(np.abs(image_data - median))
            
        elif mode == 'std':
            weight = np.std(image_data)
        else:
            raise ValueError(f"{mode=} not supported. Modes available: {weight_modes}")

    logger.info(f"Weight {weight} for {image_path}")
    return weight

def generate_weights_list_and_files(image_paths: Collection[Path], mode: str='median') -> str:
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
        weight_file = image.with_suffix(".weights")
        weight_file_list.append(weight_file)
        
        # Must be of the format:
        # #Channel Weight
        # 0 1234.5
        # 1 6789.0
        with open(weight_file, "w") as out_file:
            out_file.write("#Channel Weight\n")
            image_weight = get_image_weight(image_path=image)
            out_file.write(f"0 {image_weight}\n")

    weight_str = [str(weight_file) for weight_file in weight_file_list if weight_file.exists()]
    weight_list = "[" + ",".join(weight_str) + "]"

    return weight_list

def generate_linmos_parameter_set(
    images: Collection[Path],
    parset_output_name: Path,
    image_output_name: str = "linmos_field",
    weight_list: Optional[str] = None,
    holofile: Optional[Path] = None,
) -> Path:
    """Generate a parset file that will be used with the
    yandasoft linmos task.

    Args:
        images (Collection[Path]): The images that will be coadded into a single field image.
        parset_output_name (Path): Path of the output linmos parset file.
        image_output_name (str, optional): Name of the output image linmos produces. The weight image will have a similar name. Defaults to "linmos_field".
        weight_list (str, optional): If not None, this string will be embedded into the yandasoft linmos parset as-is. It should represent the formatted string pointing to weight files, and should be equal length of the input images. If None it is internally generated. Defaults to None.
        holofile (Optional[Path], optional): Path to a FITS cube produced by the holography processing pipeline. Used by linmos to appropriate primary-beam correct the images. Defaults to None.

    Returns:
        Path: Path to the output parset file.
    """

    img_str: List[str] = list(
        set([str(p).replace(".fits", "") for p in images if p.exists()])
    )
    logger.info(f"{len(img_str)} unique images from {len(images)} input collection. ")
    img_list: str = "[" + ",".join(img_str) + "]"

    assert len(img_str) == len(
        images
    ), "Some images were dropped from the linmos image string. Something is bad, walk the plank. "

    # If no weights_list has been provided (and therefore no optimal
    # beam-wise weighting) assume that all beams are of about the same
    # quality. In reality, this should be updated to provide a RMS noise
    # estimate per-pixel of each image.
    if weight_list is None:
        weight_list = generate_weights_list_and_files(image_paths=images, mode='median')
    
    beam_order_strs = [str(extract_beam_from_name(str(p.name))) for p in images]
    beam_order_list = "[" + ",".join(beam_order_strs) + "]"
     
    # Parameters are taken from arrakis
    parset = (
        f"linmos.names            = {img_list}\n"
        f"linmos.weights          = {weight_list}\n"
        f"linmos.beams            = {beam_order_list}\n"
        # f"linmos.beamangle        = {beam_angle_list}\n"
        f"linmos.imagetype        = fits\n"
        f"linmos.outname          = {image_output_name}_linmos\n"
        f"linmos.outweight        = {image_output_name}_weight\n"
        f"# For ASKAPsoft>1.3.0\n"
        f"linmos.useweightslog    = true\n"
        f"linmos.weighttype       = Combined\n"
        f"linmos.weightstate      = Inherent\n"
        f"linmos.cutoff           = 0.1\n"
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

    # Now write the file, me hearty
    logger.info(f"Writing parset to {str(parset_output_name)}.")
    logger.info(f"{parset}")
    assert not Path(
        parset_output_name
    ).exists(), f"The parset {parset_output_name} already exists!"
    with open(parset_output_name, "w") as parset_file:
        parset_file.write(parset)

    return parset_output_name


def linmos_images(
    images: Collection[Path],
    parset_output_name: Path,
    image_output_name: str = "linmos_field",
    weight_list: Optional[str] = None,
    holofile: Optional[Path] = None,
    container: Path = Path("yandasoft.sif"),
) -> LinmosCMD:
    """Create a linmos parset file and execute it.

    Args:
        images (Collection[Path]): The images that will be coadded into a single field image.
        parset_output_name (Path): Path of the output linmos parset file.
        image_output_name (str, optional): Name of the output image linmos produces. The weight image will have a similar name. Defaults to "linmos_field".
        weight_list (str, optional): If not None, this string will be embedded into the yandasoft linmos parset as-is. It should represent the formatted string pointing to weight files, and should be equal length of the input images. If None it is internally generated. Defaults to None.
        holofile (Optional[Path], optional): Path to a FITS cube produced by the holography processing pipeline. Used by linmos to appropriate primary-beam correct the images. Defaults to None.
        container (Path, optional): Path to the singularity container that has the yandasoft tools. Defaults to Path('yandasoft.sif').

    Returns:
        LinmosCMD: The linmos command executed and the associated parset file
    """

    assert (
        container.exists()
    ), f"The yandasoft container {str(container)} was not found. "

    linmos_parset = generate_linmos_parameter_set(
        images=images,
        parset_output_name=parset_output_name,
        image_output_name=image_output_name,
        weight_list=weight_list,
        holofile=holofile,
    )

    linmos_cmd_str = f"linmos -c {str(linmos_parset)}"
    bind_dirs = [image.absolute() for image in images] + [linmos_parset.absolute()]
    if holofile:
        bind_dirs.append(holofile.absolute())

    run_singularity_command(
        image=container, command=linmos_cmd_str, bind_dirs=bind_dirs
    )

    linmos_cmd = LinmosCMD(cmd=linmos_cmd_str, parset=linmos_parset)

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
        "parset_output_name", type=Path, help="The output path of the linmos parser"
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

    return parser


def cli() -> None:
    parser = get_parser()

    args = parser.parse_args()

    if args.mode == "parset":
        if args.yandasoft_container is None:
            generate_linmos_parameter_set(
                images=args.images,
                parset_output_name=args.parset_output_name,
                image_output_name=args.image_output_name,
                weight_list=args.weight_list,
                holofile=args.holofile,
            )
        else:
            linmos_images(
                images=args.images,
                parset_output_name=args.parset_output_name,
                image_output_name=args.image_output_name,
                weight_list=args.weight_list,
                holofile=args.holofile,
                container=args.yandasoft_container,
            )


if __name__ == "__main__":
    cli()
