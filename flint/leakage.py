"""Construct a leakge map between two polarisations, typically V/I"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, NamedTuple, Union, Optional, Tuple

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel

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
    search_box_size: int = 3
    """The size of a box to search for peak polarised signal in"""
    noise_box_size: int = 30
    """the size of a box to compute a local RMS noise measure from"""


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


class PixelCoords(NamedTuple):
    """Slim container to help collect and maintain pixel coordinates. Not intended for extensive use"""

    y: np.ndarray
    """The y-coordinate of a set of pixels"""
    x: np.ndarray
    """The x-coordinate of a set of pixels"""


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
    """Return a table given either a loaded table or a path to a table on disk"""
    component_table = (
        Table.read(catalogue) if isinstance(catalogue, Path) else catalogue
    )
    logger.info(f"Loaded component table of {len(component_table)}")

    return component_table


def filter_components(
    table: Table,
    peak_col: str,
    int_col: str,
    leakage_filters: LeakageFilters,
    ra_col: Optional[str] = None,
    dec_col: Optional[str] = None,
) -> Table:
    """Apply the pre-processing operations to catalogue components to select an
    optimal sample of sources for leakage characterisation. Sources will be selected
    based on:

    * how isolated they are
    * compactness, as traced by their int/peak

    Args:
        table (Table): Collection of sources, as produced from a source finder
        peak_col (str): The column name describing the peak flux density
        int_col (str): The column name describing integrated flux
        leakage_filters (LeakageFilters): Criteria applied to the source components in the table
        ra_col (Optional[str], optional): The RA column name. If None, it will be guessed. Defaults to None.
        dec_col (Optional[str], optional): The Dec column name. If None, it will be guessed. Defaults to None.

    Returns:
        Table: A filtered table
    """
    ra_col = guess_column_in_table(table=table, column="ra", guess_column=ra_col)
    dec_col = guess_column_in_table(table=table, column="dec", guess_column=dec_col)

    assert all(
        [col in table.colnames for col in (ra_col, dec_col, peak_col, int_col)]
    ), f"Supplied column names {ra_col=} {dec_col=} {peak_col} {int_col=} partly missing from {table.colnames}"

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
) -> PixelCoords:
    """Convert (RA, Dec) positions in a catalogue into (x, y)-pixels given an WCS

    Args:
        table (Table): The table containing sources to collect (x, y)-coodinates
        wcs (WCS): The WCS description to use to resolve (RA, Dec) to (x, y)
        ra_col (Optional[str], optional): The RA column name. If None, it will be guessed. Defaults to None.
        dec_col (Optional[str], optional): The Dec column name. If None, it will be guessed. Defaults to None.

    Returns:
        PixelCoords: _description_
    """
    ra_col = guess_column_in_table(table=table, column="ra")
    dec_col = guess_column_in_table(table=table, column="dec")

    sky_coord = SkyCoord(table[ra_col], table[dec_col], unit=(u.deg, u.deg))
    x, y = skycoord_to_pixel(wcs=wcs.celestial, coords=sky_coord, origin=0)

    return PixelCoords(y=np.floor(y).astype(int), x=np.floor(x).astype(int))


def load_and_filter_components(
    catalogue: TableOrPath, leakage_filters: LeakageFilters
) -> Table:
    """Load in a component catalogue table and apply filters to them. The
    remaining components will be used to characterise leakage

    Args:
        catalogue (TableOrPath): The path to a component catalogue, or a loaded component catalogue
        leakage_filters (LeakageFilters): Filtering options to find ideal components for leakage characterisation

    Returns:
        Table: Filtered component catalogue
    """
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


def extract_pol_stats_in_box(
    pol_image: np.ndarray,
    pixel_coords: PixelCoords,
    search_box_size: int,
    noise_box_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct two boxes around nominated pixel coordinates to:

    * extract the peak signal within
    * calculate a local RMS value for

    Args:
        pol_image (np.ndarray): The loaded polarised image
        pixel_coords (PixelCoords): Collection of pixel positioncs to evaluate the peak polarisation and noise at
        search_box_size (int): Size of box to extract the maximum polarised signal from
        noise_box_size (int): Size of box to calculate the RMS over

    Returns:
        Tuple[np.ndarray, np.ndarray]: Extracted peak polarised signal and noise
    """

    y_max, x_max = pol_image.shape[-2:]

    logger.info(f"{pol_image.shape=}, extracted {y_max=} and {x_max=}")

    pol_peak = None
    pol_noise = None
    for idx, box_size in enumerate((search_box_size, noise_box_size)):
        box_delta = int(np.ceil(box_size / 2))

        y_edge_min = np.maximum(pixel_coords.y - box_delta, 0).astype(int)
        y_edge_max = np.minimum(pixel_coords.y + box_delta, y_max - 1).astype(int)

        x_edge_min = np.maximum(pixel_coords.x - box_delta, 0).astype(int)
        x_edge_max = np.minimum(pixel_coords.x + box_delta, x_max - 1).astype(int)

        assert not np.any(y_edge_min == y_edge_max), "The y box edges are equal"
        assert not np.any(x_edge_min == x_edge_max), "The x box edges are equal"

        pol_image = np.squeeze(pol_image)

        # search_box = np.squeeze(pol_image)[y_edge_min:y_edge_max, x_edge_min:x_edge_max]
        search_box = [
            pol_image[y_min:y_max, x_min:x_max].flatten()
            for (y_min, y_max, x_min, x_max) in zip(
                y_edge_min, y_edge_max, x_edge_min, x_edge_max
            )
        ]

        if idx == 0:
            logger.info(np.nanargmax(np.abs(search_box[0])))
            pol_peak = np.array(
                [
                    (
                        data[np.nanargmax(np.abs(data))]
                        if np.any(np.isfinite(data))
                        else np.nan
                    )
                    for data in search_box
                ]
            )
        elif idx == 1:
            pol_noise = np.array([np.nanstd(data) for data in search_box])

    assert pol_peak is not None, f"{pol_peak=}, which should not happen"
    assert pol_noise is not None, f"{pol_noise=}, which should not happen"

    return pol_peak, pol_noise


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

    i_pixel_coords = get_xy_pixel_coords(table=components, wcs=i_fits.wcs)
    pol_pixel_coords = get_xy_pixel_coords(table=components, wcs=pol_fits.wcs)

    i_values = np.squeeze(i_fits.data[..., i_pixel_coords.y, i_pixel_coords.x])
    pol_peak, pol_noise = extract_pol_stats_in_box(
        pol_image=pol_fits.data,
        pixel_coords=pol_pixel_coords,
        search_box_size=leakage_filters.search_box_size,
        noise_box_size=leakage_filters.noise_box_size,
    )
    frac_values = pol_peak / i_values

    logger.info(f"{frac_values.shape=}")
    components["i_pixel_value"] = i_values
    components[f"{pol}_fraction"] = frac_values
    components[f"{pol}_peak"] = pol_peak
    components[f"{pol}_noise"] = pol_noise

    if isinstance(catalogue, Path):
        catalogue_suffix = catalogue.suffix
        output_base = (
            catalogue.with_suffix(f".{pol}_leakage.{catalogue_suffix}")
            if output_base is None
            else output_base
        )

    assert (
        output_base is not None
    ), f"{output_base=} is empty, and no catalogue path provided"

    for col in components.colnames:
        logger.info(col)
    components.write(output_base, overwrite=True)

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

    args = parser.parse_args()

    create_leakge_maps(
        stokes_i_image=args.stokes_i_image,
        pol_image=args.pol_image,
        catalogue=args.component_catalogue,
        pol=args.pol,
    )


if __name__ == "__main__":
    cli()
