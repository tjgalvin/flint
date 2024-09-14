"""Tooling related to the convolution of images. Principally
this is mostly to smooth to a common resolution
"""

from __future__ import annotations

import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Collection, List, NamedTuple, Optional

import astropy.units as u
import numpy as np
from astropy.wcs import FITSFixedWarning
from racs_tools import beamcon_2D, beamcon_3D
from radio_beam import Beam, Beams

from flint.logging import logger

warnings.simplefilter("ignore", FITSFixedWarning)


class BeamShape(NamedTuple):
    """A simple container to represent a fitted 2D gaussian,
    intended for the main lobe of the synthesised beam. This
    class has been defined to avoid issues with the serialisation
    of astropy.units, which can cause strange and wonderful
    errors when being sent over the wire to workers."""

    bmaj_arcsec: float
    """The size of the major-axis of the beam, in arcseconds."""
    bmin_arcsec: float
    """The size of the minor-axis of the beam, in arcseconds."""
    bpa_deg: float
    """Rotation of the beam, in degrees."""

    @classmethod
    def from_radio_beam(cls, radio_beam: Beam) -> BeamShape:
        """A helper function to convert a radio_beam.Beam into a
        BeamShape. This is prinicpally intended to be used when
        there is a need to exchange a Beam between processes
        that would need to serialise the object.

        Args:
            radio_beam (Beam): The Beam to convert to normalised and known units

        Returns:
            BeamShape: The normalised container without astropy units.
        """
        return cls(
            bmaj_arcsec=radio_beam.major.to(u.arcsecond).value,  # type: ignore
            bmin_arcsec=radio_beam.minor.to(u.arcsecond).value,  # type: ignore
            bpa_deg=radio_beam.pa.to(u.degree).value,  # type: ignore
        )


def get_cube_common_beam(
    cube_paths: Collection[Path], cutoff: Optional[float] = None
) -> List[BeamShape]:
    """Given a set of input cube FITS files, compute a common beam
    for each channel.

    Args:
        cube_paths (Collection[Path]): Set of cube FITS files to inspect to derive a common beam
        cutoff (Optional[float], optional): A cutoff value, in arcsec, that specifies the maximum BMAJ allowed. Defaults to None.

    Returns:
        List[BeamShape]: List of target beam shapes to use, corresponding to each channel
    """

    _, common_beam_data_list = beamcon_3D.smooth_fits_cube(
        infiles_list=list(cube_paths),
        dryrun=True,
        cutoff=cutoff,
        mode="natural",
        conv_mode="robust",
    )
    # Make proper check here that accounts for NaNs
    for file in common_beam_data_list:
        assert all(
            (file[0].major == common_beam_data_list[0][0].major)
            | np.isnan(file[0].major)
        )
        assert all(
            (file[0].minor == common_beam_data_list[0][0].minor)
            | np.isnan(file[0].minor)
        )
        assert all(
            (file[0].pa == common_beam_data_list[0][0].pa) | np.isnan(file[0].pa)
        )

    first_cube_fits_beam = common_beam_data_list[0][0]
    assert isinstance(
        first_cube_fits_beam, Beams
    ), f"Unexpected type for common beams. Expected Beams, got {type(first_cube_fits_beam)}"

    beam_shape_list = [
        BeamShape(
            bmaj_arcsec=beam.major.to("arcsec").value,  # type: ignore
            bmin_arcsec=beam.minor.to("arcsec").value,  # type: ignore
            bpa_deg=beam.pa.to("degree").value,  # type: ignore
        )
        for beam in first_cube_fits_beam
    ]
    return beam_shape_list


def get_common_beam(
    image_paths: Collection[Path], cutoff: Optional[float] = None
) -> BeamShape:
    """Return the minimum beam size required to encompass the beams described
    in the FITS header (e.g. BMAJ,BMIN,BPA) of the input images. This is used
    when preparing to convolve to a common resolution.

    Args:
        image_paths (Collection[Path]): The paths to the FITS images that will be examined.
        cutoff (Optional[float], optional): The maximum beam size an input image is allowed to have. BMAJ's larger than this are ignored from the calculation. Defaults to None.

    Returns:
        BeamShape: Smalled common beam available to be used
    """

    logger.info(f"Calculating common beam size of {len(image_paths)} images. ")

    if cutoff:
        logger.info(f"Setting beam cutoff to {cutoff} arcseconds. ")

    beam, beams = beamcon_2D.get_common_beam(files=list(image_paths), cutoff=cutoff)

    beam_shape = BeamShape.from_radio_beam(beam)
    logger.info(f"Constructed {beam_shape=}")

    return beam_shape


def convolve_images(
    image_paths: Collection[Path],
    beam_shape: BeamShape,
    cutoff: Optional[float] = None,
    convol_suffix: str = "conv",
) -> Collection[Path]:
    """Convolve a set of input images to a common resolution as specified
    by the beam_shape. If the major-axis of the native resolution is larger
    than cutoff (in arcseconds) then the racs_tools beamconv_2D task will
    nan it.

    Args:
        image_paths (Collection[Path]): Set of image paths to FITS images to convol
        beam_shape (BeamShape): The specification of the desired final resolution
        cutoff (Optional[float], optional): Images whose major-axis is larger than this will be blank. Expected in arcseconds. Defaults to None.
        convol_suffix (str, optional): The suffix added to .fits to indicate smoothed image. Defaults to 'conv'.

    Returns:
        Collection[Path]: Set of paths to the smoothed images
    """

    logger.info(f"Will attempt to convolve {len(image_paths)} images.")
    if cutoff:
        logger.info(f"Supplied cutoff of {cutoff} arcsecond")

    radio_beam = Beam(
        major=beam_shape.bmaj_arcsec * u.arcsecond,
        minor=beam_shape.bmin_arcsec * u.arcsecond,
        pa=beam_shape.bpa_deg * u.deg,
    )

    conv_image_paths: List[Path] = []

    for image_path in image_paths:
        logger.info(f"Convolving {str(image_path.name)}")
        beamcon_2D.beamcon_2d_on_fits(
            file=image_path,
            outdir=None,
            new_beam=radio_beam,
            conv_mode="robust",
            suffix=convol_suffix,
            cutoff=cutoff,
        )
        conv_image_paths.append(
            Path(str(image_path).replace(".fits", f".{convol_suffix}.fits"))
        )

    return conv_image_paths


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)

    subparsers = parser.add_subparsers(dest="mode")

    convol_parser = subparsers.add_parser(
        "convol", help="Convol images to a common resolution"
    )

    convol_parser.add_argument(
        "images",
        type=Path,
        nargs="+",
        help="The images that will be convolved to a common resolution",
    )
    convol_parser.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="Beams whose major-axis are larger then this (in arcseconds) are ignored from the calculation of the optimal beam.",
    )
    convol_parser.add_argument(
        "--convol-suffix",
        type=str,
        default="conv",
        help="The suffix added to convolved images. ",
    )

    maxbeam_parser = subparsers.add_parser(
        "maxbeam", help="Find the optimal beam size for a set of images."
    )

    maxbeam_parser.add_argument(
        "images",
        type=Path,
        nargs="+",
        help="The images that will be convolved to a common resolution",
    )
    maxbeam_parser.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="Beams whose major-axis are larger then this (in arcseconds) are ignored from the calculation of the optimal beam.",
    )

    cubemaxbeams_parser = subparsers.add_parser(
        "cubemaxbeam",
        help="Calculate the set of common beams across channels in a set of cubes",
    )
    cubemaxbeams_parser.add_argument(
        "cubes",
        type=Path,
        nargs="+",
        help="The images that will be convolved to a common resolution",
    )
    cubemaxbeams_parser.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="Beams whose major-axis are larger then this (in arcseconds) are ignored from the calculation of the optimal beam.",
    )

    return parser


def cli() -> None:
    parser = get_parser()

    args = parser.parse_args()

    if args.mode == "maxbeam":
        get_common_beam(image_paths=args.images, cutoff=args.cutoff)
    if args.mode == "convol":
        common_beam = get_common_beam(image_paths=args.images, cutoff=args.cutoff)
        _ = convolve_images(
            image_paths=args.images,
            beam_shape=common_beam,
            cutoff=args.cutoff,
            convol_suffix=args.convol_suffix,
        )
    if args.mode == "cubemaxbeam":
        common_beam_shape_list = get_cube_common_beam(
            cube_paths=args.cubes, cutoff=args.cutoff
        )
        logger.info(f"Extracted {common_beam_shape_list=}")


if __name__ == "__main__":
    cli()
