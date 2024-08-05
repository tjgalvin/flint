"""Construct a leakge map between two polarisations, typically V/I"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, NamedTuple, Union, Optional

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from flint.catalogue import guess_column_in_table
from flint.logging import logger


TableOrPath = Union[Table, Path]


class LeakageFilters(NamedTuple):
    """Description of the filtering options to apply to components
    when characterising leakage"""

    isolation_radius_deg: float = 0.0055
    """The minimum distance to the nearest component"""
    upper_int_peak_ratio: float = 2.0
    """The upper limit on acceptable int/peak ratios"""
    lower_int_peak_ratio: float = 0.5
    """The lower limit on acceptable int/peak ratios"""


class FITSImage(NamedTuple):
    """Container to couple FITS header, image and WCS"""

    data: np.ndarray
    """The data of the fits image"""
    header: Dict
    """Header of the fits image"""
    wcs: WCS
    """Celestial WCS of the fits image"""
    path: Path
    """Path of the loaded FITS image on disk"""


def load_fits_image(fits_path: Path) -> FITSImage:
    """Load in a FITS image and package the components into a consistent
    form

    Args:
        fits_path (Path): The path of the FITS image to examining

    Returns:
        FITSImage: Loaded FITS properties
    """

    assert (
        fits_path.suffix == ".fits"
    ), f"Unexpected file type for {fits_path=}, expected fits"
    logger.info(f"Opening {fits_path=}")
    with fits.open(fits_path) as in_fits:
        image_data = in_fits[0].data
        header = dict(in_fits[0].header.items())
        wcs = WCS(header)

    return FITSImage(data=image_data, header=header, wcs=wcs, path=fits_path)


def _load_component_table(catalogue: TableOrPath) -> Table:
    component_table = (
        Table.read(catalogue) if isinstance(catalogue, Path) else catalogue
    )
    logger.info(f"Loaded component table of {len(component_table)}")

    return component_table


def filter_components(
    table: Table,
    ra_col: str,
    dec_col: str,
    peak_col: str,
    int_col: str,
    leakage_filters: LeakageFilters,
) -> Table:
    assert all(
        [col in table.colnames for col in (ra_col, dec_col, peak_col, int_col)]
    ), f"Supplied column names partly missing from {table.colnames}"

    total_comps = len(table)
    sky_coords = SkyCoord(table[ra_col], table[dec_col], unit=(u.hour, u.deg))

    # The match_to_catalog_sky return idx, sep2d, sep3d. We care about separation, matey
    isolation_mask = sky_coords.match_to_catalog_sky(sky_coords, nthneighbor=2)[1] > (
        leakage_filters.isolation_radius_deg * u.deg
    )  # type: ignore
    logger.info(f"{np.sum(isolation_mask)} of {total_comps} sources are isolated")

    ratio = table[int_col] / table[peak_col]  # type: ignore
    ratio_mask = (leakage_filters.lower_int_peak_ratio < ratio) & (
        ratio < leakage_filters.upper_int_peak_ratio
    )  # type: ignore
    logger.info(f"{np.sum(ratio_mask)} of {total_comps} sources are compact")

    mask = isolation_mask & ratio_mask
    logger.info(f"{np.sum(mask)} of {total_comps} sources are isolated and are compact")

    table = table[mask]  # type: ignore

    return table


def get_xy_pixel_coords(
    table: Table,
    wcs: WCS,
    ra_col: Optional[str] = None,
    dec_col: Optional[str] = None,
) -> np.ndarray:
    ra_col = guess_column_in_table(table=table, column="ra")
    dec_col = guess_column_in_table(table=table, column="dec")

    sky_coord = SkyCoord(table[ra_col], table[dec_col], unit=(u.hour, u.deg))
    y, x = wcs.all_world2pix(sky_coord)
    pixel_coords = np.array((y, x))

    return pixel_coords


def load_and_filter_components(
    catalogue: TableOrPath, leakage_filters: LeakageFilters
) -> Table:
    comp_table = _load_component_table(catalogue=catalogue)
    ra_col = guess_column_in_table(table=comp_table, column="ra")
    dec_col = guess_column_in_table(table=comp_table, column="dec")
    peak_col = guess_column_in_table(table=comp_table, column="peakflux")
    int_col = guess_column_in_table(table=comp_table, column="intflux")

    comp_table = filter_components(
        table=comp_table,
        ra_col=ra_col,
        dec_col=dec_col,
        peak_col=peak_col,
        int_col=int_col,
        leakage_filters=leakage_filters,
    )
    return comp_table


def create_leakge_maps(
    stokes_i_image: Path,
    pol_image: Path,
    catalogue: Union[Table, Path],
    pol: str = "v",
    output_base: Optional[Path] = None,
) -> Path:
    i_fits = load_fits_image(fits_path=stokes_i_image)
    pol_fits = load_fits_image(fits_path=pol_image)

    leakage_filters = LeakageFilters()

    components = load_and_filter_components(
        catalogue=catalogue, leakage_filters=leakage_filters
    )

    logger.info(f"{i_fits} {pol_fits} {components}")

    i_pixel_coords = get_xy_pixel_coords(table=components, wcs=i_fits.wcs)
    pol_pixel_coords = get_xy_pixel_coords(table=components, wcs=pol_fits.wcs)

    i_values = i_fits.data[..., i_pixel_coords]
    pol_values = pol_fits.data[..., pol_pixel_coords]
    frac_values = i_values / pol_values

    logger.info(f"{frac_values.shape=}")
    components["i_pixel_value"] = i_values
    components[f"{pol}_fraction"] = frac_values
    components[f"{[pol]}_pixel_value"] = i_values

    if isinstance(catalogue, Path):
        catalogue_suffix = catalogue.suffix
        output_base = (
            catalogue.with_suffix(f"{pol}_leakage.{catalogue_suffix}")
            if output_base is None
            else output_base
        )

    assert (
        output_base is not None
    ), f"{output_base=} is empty, and no catalogue path provided"

    return stokes_i_image


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Create a leakage cataloge and map")
    parser.add_argument("stokes_i_image", type=Path, help="Path to the stokes-i image")
    parser.add_argument("pol_image", type=Path, help="Path to the polarisation image")
    parser.add_argument(
        "component_catalogue", type=Path, help="Path to the component catalogue"
    )
    parser.add_argument(
        "--pol",
        default="v",
        choices=("q", "u", "v"),
        help="The polarisation that is being considered",
    )

    return parser


def cli() -> None:
    parser = get_parser()

    _ = parser.parse_args()


if __name__ == "__main__":
    cli()
