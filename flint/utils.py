"""Collection of functions and tooling intended
for general usage.
"""

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from flint.logging import logger


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


def generate_stub_wcs_header(
    ra: float,
    dec: float,
    image_shape: Tuple[int, int],
    pixel_scale: Union[u.Quantity, str, float],
    projection: str = "SIN",
    base_wcs: Optional[Union[Path, WCS]] = None,
) -> WCS:
    """Create a basic WSC header object that can be used to calculate sky positions
    for an example image.

    Args:
        ra (float): The RA in degrees at the reference position
        dec (float): The Dec in degrees at the reference position
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
    if isinstance(pixel_scale, str):
        pixel_scale = u.Quantity(pixel_scale)
    elif isinstance(pixel_scale, float):
        pixel_scale = pixel_scale * u.arcsec

    # Trust nothing even more
    assert isinstance(
        pixel_scale, u.Quantity
    ), f"pixel_scale is not an quantity, instead {type(pixel_scale)}"
    pixel_scale = np.abs(pixel_scale.to(u.rad).value)

    image_center = np.array(image_shape, dtype=int) // 2

    # Sort out the header. If Path get the header through and construct the WCS
    if isinstance(base_wcs, Path):
        base_wcs = WCS(fits.getheader(base_wcs))

    # The celestial guarantees only two axis
    w = base_wcs.celestial if base_wcs else WCS(naxis=2)

    # Nor bring it all together
    w.wcs.crpix = image_center
    w.wcs.cdelt = np.array([-pixel_scale, pixel_scale])
    w.wcs.crval = [ra, dec]
    w.wcs.ctype = [f"RA---{projection}", f"DEC--{projection}"]
    w.wcs.cunit = ["deg", "deg"]
    w._naxis1 = image_shape[0]
    w._naxis2 = image_shape[1]

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


def create_directory(directory: Path) -> Path:
    """Will attempt to safely create a directory. Should it
    not exist it will be created. if this creates an exception,
    which might happen in a multi-process environment, it is
    reported and discarded.

    Args:
        directory (Path): Path to directory to create

    Returns:
        Path: The directory created
    """

    directory = Path(directory)

    if directory.exists():
        return directory

    logger.info(f"Creating {str(directory)}")
    try:
        directory.mkdir(parents=True)
    except Exception as e:
        logger.error(f"Failed to create {str(directory)} {e}.")

    return directory
