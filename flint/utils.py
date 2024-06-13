"""Collection of functions and tooling intended
for general usage.
"""

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Union

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from flint.convol import BeamShape
from flint.logging import logger

# TODO: This Captain is aware that there is a common fits getheader between
# a couple of functions that interact with tasks. Perhaps a common FITS properties
# struct should be considered. The the astropy.io.fits.Header might be
# appropriate to pass around between dask / prefect delayed functions. Something
# that only opens the FITS file once and places things into common field names.


def get_beam_shape(fits_path: Path) -> Optional[BeamShape]:
    """Construct and return a beam shape from the fields in a FITS image

    Args:
        fits_path (Path): FITS image to extract the beam information from

    Returns:
        Optional[BeamShape]: Shape of the beam stored in the FITS image. None is returned if the beam is not found.
    """

    header = fits.getheader(filename=fits_path)

    if not all([key in header for key in ("BMAJ", "BMIN", "BPA")]):
        return None

    beam_shape = BeamShape(
        bmaj_arcsec=header["BMAJ"] * 3600,
        bmin_arcsec=header["BMIN"] * 3600,
        bpa_deg=header["BPA"],
    )

    return beam_shape


def get_pixels_per_beam(fits_path: Path) -> Optional[float]:
    """Given a image with beam information, return the number of pixels
    per beam. The beam is taken from the FITS header. This is evaluated
    for pixels at the reference pixel position.

    Args:
        fits_path (Path): FITS iamge to consideer

    Returns:
        Optional[float]: Number of pixels per beam. If beam is not in header then None is returned.
    """

    beam_shape = get_beam_shape(fits_path=fits_path)

    if beam_shape is None:
        return None

    header = fits.getheader(filename=fits_path)

    pixel_ra = np.abs(header["CDELT1"] * 3600)
    pixel_dec = np.abs(header["CDELT2"] * 3600)

    beam_area = beam_shape.bmaj_arcsec * beam_shape.bmin_arcsec * np.pi
    pixel_area = pixel_ra * pixel_dec

    no_pixels = beam_area / pixel_area

    return no_pixels


def get_packaged_resource_path(package: str, filename: str) -> Path:
    """Load in the path of a package sources.

    The `package` argument is passed as a though the module
    is being speficied as an import statement: `flint.data.aoflagger`.

    Args:
        package (str): The module path to the resources
        filename (str): Filename of the datafile to load

    Returns:
        Path: The absolute path to the packaged resource file
    """
    logger.info(f"Loading {package=} for {filename=}")
    try:
        import importlib_resources as importlib_resources
    except ImportWarning:
        from importlib import resources as importlib_resources

    with importlib_resources.files(package) as p:
        full_path = Path(p) / filename

    logger.debug(f"Resolved {full_path=}")

    return full_path


def generate_strict_stub_wcs_header(
    position_at_image_center: SkyCoord,
    image_shape: Tuple[int, int],
    pixel_scale: Union[u.Quantity, str],
    image_shape_is_center: bool = False,
) -> WCS:
    """Create a WCS object using some strict quantities. There
    are no attempts to cast values appropriately, exception being
    calling `astropy.units.Quantity` on the `pixel_scale` input
    should it not be a quantity.

    The supplied `image_size` is used to calculate the center of
    the image and set the reference pixel value.

    The output projection is SIN.

    Args:
        position_at_image_center (SkyCoord): The position that will be at the reference pixel
        image_shape (Tuple[int, int]): The size of the image
        pixel_scale (Union[u.Quantity,str]): Size of the square pixels. If `str` passed will be cast to `Quantity`.
        image_shape_is_center (bool, optional): It True the position specified by `image_shape` is the center reference position. if False, `image_shape` is assumed to be the size of the image, and teh center is computed from this. Defaults to False.

    Raises:
        TypeError: Raised when pixel scale it not a str or astropy.units.Quantity

    Returns:
        WCS: A WCS header matching the input specs
    """

    if isinstance(pixel_scale, str):
        pixel_scale = u.Quantity(pixel_scale)
    elif not isinstance(pixel_scale, u.Quantity):
        raise TypeError(
            f"pixel_scale should be of type astro.units.Quantity or str, got {type(pixel_scale)}"
        )

    # This should be good enough
    image_center = image_shape
    if not image_shape_is_center:
        image_center = np.array(image_center) / 2
        logger.debug(f"Constructed WCS {image_center=}")

    header = {
        "CRVAL1": position_at_image_center.ra.deg,
        "CRVAL2": position_at_image_center.dec.deg,
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CDELT1": -pixel_scale.to(u.deg).value,
        "CDELT2": pixel_scale.to(u.deg).value,
        "CRPIX1": image_center[0],
        "CRPIX2": image_center[1],
        "CTYPE1": "RA---SIN",
        "CTYPE2": "DEC--SIN",
        "SPECSYS": "TOPOCENT",
    }

    wcs = WCS(fits.Header(header))

    return wcs


def generate_stub_wcs_header(
    ra: Optional[Union[float, u.Quantity]] = None,
    dec: Optional[Union[float, u.Quantity]] = None,
    image_shape: Optional[Tuple[int, int]] = None,
    pixel_scale: Optional[Union[u.Quantity, str, float]] = None,
    projection: str = "SIN",
    base_wcs: Optional[Union[Path, WCS]] = None,
) -> WCS:
    """Create a basic WSC header object that can be used to calculate sky positions
    for an example image.

    Care should be taken when using this function as it tries to be too
    smart for its own good.

    Args:
        ra (fUnion[loat,u.Quantuty]): The RA at the reference pixel. if a float is provided it is assumed to be in degrees.
        dec (Union[float,u.Quantuty]): The Dec at the reference pizel. if a float is provided it is assumed to be in degrees.
        image_shape (Tuple[int, int]): Size of the representative image
        pixel_scale (Union[u.Quantity, str, float]): The size of the square pixels. if a `float` it is assumed to be arcseconds. If `str`, parsing is hangled by `astropy.units.Quantity`.
        projection (str, optional): Project scheme to encode in the header. Defaults to "SIN".
        base_wcs (Optional[Union[Path, WCS]], optional): Overload an existing WCS object with argument properties. If a `Path` the WCS is obtained from the fits file. If `None` WCS is built from arguments. Defaults to None.

    Returns:
        WCS: The representative WCS objects
    """
    # Trust nothing
    assert (
        len(projection) == 3
    ), f"Projection should be three characters, received {projection}"

    # Handle all the pixels you rotten seadog
    if pixel_scale is not None:
        if isinstance(pixel_scale, str):
            pixel_scale = u.Quantity(pixel_scale)
        elif isinstance(pixel_scale, float):
            pixel_scale = pixel_scale * u.arcsec

        # Trust nothing even more
        assert isinstance(
            pixel_scale, u.Quantity
        ), f"pixel_scale is not an quantity, instead {type(pixel_scale)}"
        pixel_scale = np.abs(pixel_scale.to(u.rad).value)

        pixel_scale = np.array([-pixel_scale, pixel_scale])

    # Handle the ref position
    if ra is not None:
        ra = ra if isinstance(ra, u.Quantity) else ra * u.deg
    if dec is not None:
        dec = dec if isinstance(dec, u.Quantity) else dec * u.deg

    # Sort out the header. If Path get the header through and construct the WCS
    if base_wcs is not None:
        if isinstance(base_wcs, Path):
            base_wcs = WCS(fits.getheader(base_wcs)).celestial

        assert isinstance(
            base_wcs, WCS
        ), f"Expecting base_wcs to be a WCS object by now, instead is {type(base_wcs)}"

        if image_shape is None:
            image_shape = base_wcs._naxis
        if ra is None:
            ra = base_wcs.wcs.crval[0] * u.Unit(base_wcs.wcs.cunit[0])
        if dec is None:
            dec = base_wcs.wcs.crval[1] * u.Unit(base_wcs.wcs.cunit[1])
        if pixel_scale is None:
            pixel_scale = base_wcs.wcs.cdelt

    # Only needs to be approx correct. Off by one pixel should be ok, this pirate thinks
    if image_shape is not None:
        image_center = np.array(image_shape, dtype=int) // 2

    # The celestial guarantees only two axis
    w = base_wcs.celestial if base_wcs else WCS(naxis=2)

    if any([i is None for i in (image_shape, ra, dec, pixel_scale)]):
        raise ValueError("Something is unset, and unable to form wcs object. ")

    # Nor bring it all together
    w.wcs.crpix = image_center
    w.wcs.cdelt = pixel_scale
    w.wcs.crval = [ra.to(u.deg).value, dec.to(u.deg).value]
    w.wcs.ctype = [f"RA---{projection}", f"DEC--{projection}"]
    w.wcs.cunit = ["deg", "deg"]
    w._naxis = tuple(image_shape)

    return w


def estimate_skycoord_centre(
    sky_positions: SkyCoord, final_frame: str = "fk5"
) -> SkyCoord:
    """Estimate the centre position of (RA, Dec) positions stored in a
    input `SkyCoord`. Internally the mean of the cartesian (X,Y,Z) is
    calculated, which is then transormed back to a sky position,.

    Args:
        sky_positions (SkyCoord): Set of input positions to consider
        final_frame (str, optional): The frame of the returned `SkyCoord` objects set using `.transform_to`. Defaults to fk5.

    Returns:
        SkyCoord: Centre position
    """
    xyz_positions = sky_positions.cartesian.xyz
    xyz_mean_position = np.mean(xyz_positions, axis=1)

    mean_position = SkyCoord(
        *xyz_mean_position, representation_type="cartesian"
    ).transform_to(final_frame)

    return mean_position


def estimate_image_centre(image_path: Path) -> SkyCoord:
    with fits.open(image_path, memmap=True) as open_image:
        image_header = open_image[0].header
        image_shape = open_image[0].data.shape

    wcs = WCS(image_header)
    centre_pixel = np.array(image_shape) / 2.0
    # The celestial deals with the radio image potentially having four dimensions
    # (stokes, frequencyes, ra, dec)
    centre_sky = wcs.celestial.pixel_to_world(centre_pixel[0], centre_pixel[1])

    return centre_sky


def zip_folder(in_path: Path, out_zip: Optional[Path] = None) -> Path:
    """Zip a directory and remove the original.

    Args:
        in_path (Path): The path that will be zipped up.
        out_zip (Path, optional): Name of the output file. A zip extension will be added. Defaults to None.

    Returns:
        Path: the path of the compressed zipped folder
    """

    out_zip = in_path if out_zip is None else out_zip

    logger.info(f"Zipping {in_path}.")
    shutil.make_archive(str(out_zip), "zip", base_dir=str(in_path))
    remove_files_folders(in_path)

    return out_zip


def rsync_copy_directory(target_path: Path, out_path: Path) -> Path:
    """A small attempt to rsync a directtory from one location to another.
    This is an attempt to verify a copy was completed successfully.

    Args:
        target_path (Path): The target directory to copy
        out_path (Path): The location to copy the directory to

    Returns:
        Path: The output path of the new directory.
    """

    rsync_cmd = (
        f"rsync -avh --progress --stats " f"{str(target_path)}/ " f"{str(out_path)}/ "
    )
    logger.info(f"Rsync copying {target_path} to {out_path}.")
    logger.debug(f"Will run {rsync_cmd}")
    rsync_run = subprocess.Popen(rsync_cmd.split(), stdout=subprocess.PIPE)

    if rsync_run.stdout is not None:
        for line in rsync_run.stdout:
            logger.debug(line.decode().rstrip())

    return out_path


def copy_directory(
    input_directory: Path,
    output_directory: Path,
    verify: bool = False,
    overwrite: bool = False,
) -> Path:
    """Copy a directory into a new location.

    Args:
        input_directory (Path): The source directory to copy
        output_directory (Path): The location of the source directory to copy to
        verify (bool, optional): Attempt to run `rsync` to verify copy worked. Defaults to False.
        overwrite (bool, optional): Remove the target direcrtory if it exists. Defaults to False.

    Returns:
        Path: Location of output directory
    """

    input_directory = Path(input_directory)
    output_directory = Path(output_directory)

    assert (
        input_directory.exists() and input_directory.is_dir()
    ), f"Currently only supprts copying directories, {input_directory=} is a file or does not exist. "

    logger.info(f"Copying {input_directory} to {output_directory}.")

    if output_directory.exists():
        if overwrite:
            logger.warning(f"{output_directory} already exists. Removing it. ")
            remove_files_folders(output_directory)

    shutil.copytree(input_directory, output_directory)

    if verify:
        rsync_copy_directory(input_directory, output_directory)

    return output_directory


def remove_files_folders(*paths_to_remove: Path) -> List[Path]:
    """Will remove a set of paths from the file system. If a Path points
    to a folder, it will be recursively removed. Otherwise it is simply
    unlinked.

    Args:
        paths_to_remove (Path): Set of Paths that will be removed

    Returns:
        List[Path]: Set of Paths that were removed
    """

    files_removed = []

    file: Path
    for file in paths_to_remove:
        file = Path(file)
        if not file.exists():
            logger.debug(f"{file} does not exist. Skipping, ")
            continue

        if file.is_dir():
            logger.info(f"Removing folder {str(file)}")
            shutil.rmtree(file)
        else:
            logger.info(f"Removing file {file}.")
            file.unlink()

        files_removed.append(file)

    return files_removed


def create_directory(directory: Path, parents: bool = True) -> Path:
    """Will attempt to safely create a directory. Should it
    not exist it will be created. if this creates an exception,
    which might happen in a multi-process environment, it is
    reported and discarded.

    Args:
        directory (Path): Path to directory to create
        parents (bool, optional): Create parent directories if necessary. Defaults to True.

    Returns:
        Path: The directory created
    """

    directory = Path(directory)

    logger.info(f"Creating {str(directory)}")
    try:
        directory.mkdir(parents=parents, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create {str(directory)} {e}.")

    return directory
