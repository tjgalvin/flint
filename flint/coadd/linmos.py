"""This is an interface into the yandasoft linmos task. 
"""
from pathlib import Path
from typing import Optional, Collection, List
from argparse import ArgumentParser

from flint.logging import logger 


def generate_linmos_parameter_set(
    images: Collection[Path], 
    parset_output_name: Path,
    image_output_name: str = "linmos_field",
    weight_list: Optional[Path]=None,
    holofile: Optional[Path]=None
) -> Path:
    """Generate a parset file that will be used with the
    yandasoft linmos task. 

    Args:
        images (Collection[Path]): The images that will be coadded into a single field image. 
        parset_output_name (Path): Path of the output linmos parset file. 
        image_output_name (str, optional): Name of the output image linmos produces. The weight image will have a similar name. Defaults to "linmos_field".
        weight_list (Optional[Path], optional): If not None, this should be a new-line delimited text file used to weight the input images. Defaults to None.
        holofile (Optional[Path], optional): Path to a FITS cube produced by the holography processing pipeline. Used by linmos to appropriate primary-beam correct the images. Defaults to None.

    Returns:
        Path: Path to the output parset file. 
    """

    img_str: List[str] = list(set([str(p).replace(".fits","") for p in images if p.exists()]))
    logger.info(f"{len(img_str)} unique images from {len(images)} input collection. ")
    img_list: str = "[" + ",".join(img_str) + "]"
    
    assert len(img_str) == len(images), "Some images were dropped from the linmos image string. Something is bad, walk the plank. "
    
    # If no weights_list has been provided (and therefore no opttimal
    # beam-wise weighting) assume that all beams are of about the same
    # quality. In reality, this should be updated to provide a RMS noise
    # estimate per-pixel of each image. 
    # TODO: build out functionality to create RMS images, and use those
    # here
    if weight_list is None:
        weight_list = Path(f"{image_output_name}.weight_list")
        logger.warn(f"No weight list has been provided. Assuming equal weights, and writing to {weight_list}.")
        with open(weight_list, 'w') as weight_file:
            for _ in images:
                weight_file.write("1.0\n")
    
    # Parameters are taken from arrakis
    parset = (
        f"linmos.names            = {img_list}\n"
         f"linmos.imagetype        = fits\n"
         f"linmos.outname          = {image_output_name}linmos\n"
         f"linmos.outweight        = {image_output_name}weight\n"
         f"# For ASKAPsoft>1.3.0\n"
         f"linmos.useweightslog    = true\n"
         f"linmos.weighttype       = FromPrimaryBeamModel\n"
         f"linmos.weightstate      = Inherent\n"
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
            f"linmos.removeleakage    = true\n"
        )


    # Now write the file, me hearty
    logger.info(f"Writing parset to {str(parset_output_name)}.")
    with open(parset_output_name, 'w') as parset_file:
        parset_file.write(parset)

    return parset_output_name

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    
    subparsers = parser.add_subparsers(dest='mode')

    parset_parser = subparsers.add_parser('parset', help="Generate a yandasoft linmos parset")

    parset_parser.add_argument('images', type=Path, nargs='+', help='The images that will be coadded')
    parset_parser.add_argument('parset_output_name', type=Path, help='The output path of the linmos parser')
    parset_parser.add_argument('--image-output-name', type=str, default='linmos_field', help='The base name used to create the output linmos images and weight maps')
    parset_parser.add_argument('--weight-list', type=Path, default=None, help='Path a new-line delimited text-file containing the relative weights corresponding to the input images')
    parset_parser.add_argument('--holofile', type=Path, default=None, help='Path to the holography FITS cube used for primary beam corrections')

    return parser

def cli() -> None:
    parser = get_parser()

    args = parser.parse_args()
    
    if args.mode == 'parset':
        generate_linmos_parameter_set(
            images=args.images,
            parset_output_name=args.parset_output_name,
            image_output_name=args.image_output_name,
            weight_list=args.weight_list,
            holofile=args.holofile
        )


if __name__ == '__main__':
    cli()
