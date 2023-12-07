"""Utility function to create a one figure multi panel validation plot
for continuum imaging of RACS data
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pkg_resources
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from flint.logging import logger

F_SMALL = 7
F_MED = 8


class Catalogue(NamedTuple):
    """A basic structure used to describe a known catalogue."""

    survey: str
    """Shorthand name of the sourcey catalogue"""
    file_name: str
    """The file name of the known catalogue"""
    freq: float  # Hertz
    """Reference frequency of the catalogue, in Hertz"""
    ra_col: str
    """Column name containing the right-ascension"""
    dec_col: str
    """Column name containing the declination"""
    name_col: str
    """Column name containing the source/component name"""
    flux_col: str
    """Column name containing the flux density"""
    maj_col: str
    """Column name containing the major-axis of the source gaussian component"""
    min_col: str
    """Column name containing the min-axis of the source gaussian component"""
    pa_col: str
    """Column name containing the pa of the source gaussian component"""
    alpha_col: Optional[str] = None  # Used to scale the SED
    """Column name containing the spectral index, used to calculate the source SED. If None a default is used. """
    q_col: Optional[str] = None  # Used to scale the SED
    """Column name containing the curvature of the spectral index, used to calculate the source SED. If None a default is used. """


class ValidatorLayout(NamedTuple):
    """Simple container for all the matplotlib axes objects"""

    ax_rms: plt.Axes
    """Axes for the RMS of the field"""
    ax_legend: plt.Axes
    """Container for basic SBID information"""
    ax_psf: plt.Axes
    """Axes for the PSF of the image"""
    ax_counts: plt.Axes
    """Axes for quick look source counts"""
    ax_brightness1: plt.Axes
    """Axes to compare brightness of sources from the first catalogue"""
    ax_brightness2: plt.Axes
    """Axes to compare brightness of sources from the first catalogue"""
    ax_astrometry: plt.Axes
    """Axes to compare astrometry of sources"""
    ax_astrometry1: plt.Axes
    """Axes to compare astrometry of sources from the first catalogue"""
    ax_astrometry2: plt.Axes
    """Axes to compare astromnetry of sources from the first catalogue"""


class RMSImageInfo(NamedTuple):
    """Class to hold basic RMS Image information, excluding the actual raw image data"""

    path: Path
    """Path to the RMS fits image"""
    header: fits.Header
    """Header from the FITS image"""
    shape: Tuple[int, int]
    """Dimension of the image"""
    no_valid_pixels: int
    """Number of valid pixels in the image"""
    area: float
    """The area of valid sky, in degrees squared"""


class SourceCounts(NamedTuple):
    """A small container to pass around source count information"""

    bins: np.ndarray
    """The bin edges of the source counts in Jy"""
    bin_center: np.ndarray
    """The bin centers to use in Jy"""
    counts_per_bin: np.ndarray
    """The counts of sources per flux bin"""
    counts_per_bin_err: np.ndarray
    """The rough estimate on the errors per bin"""
    euclid_counts: np.ndarray
    """Euclidean normalised source counts"""
    euclid_counts_err: np.ndarray
    """Rough estimate of error on the euclidean normalised source counts"""


class MatchResult(NamedTuple):
    """Simple container to hold results of a quick cross match"""

    name1: str
    """Name of the first survey"""
    name2: str
    """Name of the second survey"""
    pos1: SkyCoord
    """Positions of sources in the first survey"""
    pos2: SkyCoord
    """Positions of sources in the second survey"""
    freq1: float
    """Frequency in Hz of the first survey"""
    freq2: float
    """Frequency in Hz of the second survey"""
    flux1: Optional[np.ndarray] = None
    """Brightness in Jy of source in the first survey"""
    flux2: Optional[np.ndarray] = None
    """Brightness in Jy of source in the second survey"""


def get_known_catalogue_info(name: str) -> Catalogue:
    """Return the parameters of a recognised catalogue.

    These are currently hardcoded.

    The structure returneed outlines the name of columns of interest:
    - RA
    - Dec
    - Integrated flux
    - Major / Minor / PA

    Args:
        name (str): The survey name of interest

    Raises:
        ValueError: Raised when an unrecongised catalogue is provided

    Returns:
        Catalogue: Information of the survey catalogue
    """
    # TODO: This should be expanded to include the units as well
    # TODO: Catalogues here need to be packaged somehow

    name = name.upper()

    if name == "NVSS":
        catalogue = Catalogue(
            survey="NVSS",
            file_name="VIII_65_nvss.dat_CH_2.fits",
            name_col="NVSS",
            freq=1.4e9,
            ra_col="RA",
            dec_col="Dec",
            flux_col="S1.4",
            maj_col="MajAxis",
            min_col="MinAxis",
            pa_col="PA",
        )
    elif name == "SUMSS":
        catalogue = Catalogue(
            survey="SUMSS",
            file_name="sumsscat.Mar-11-2008_CLH.fits",
            freq=8.43e8,
            ra_col="RA",
            dec_col="DEC",
            name_col="Mosaic",
            flux_col="St",
            maj_col="dMajAxis",
            min_col="dMinAxis",
            pa_col="dPA",
        )
    elif name == "ICRF":
        catalogue = Catalogue(
            survey="ICRF",
            file_name="icrf.csv",
            freq=1e9,
            ra_col="RA",
            dec_col="Dec",
            name_col="IERS Des.",
            flux_col="None",
            maj_col="None",
            min_col="None",
            pa_col="None",
        )
    else:
        raise ValueError(f"Unknown catalogue {name}")

    return catalogue


def load_known_catalogue(
    name: str, reference_catalogue_directory: Path
) -> Tuple[Table, Catalogue]:
    """Load in a known catalogue table

    Args:
        name (str): Name of the survey to load
        reference_catalogue_directory (Path): Tje directory location with the reference catalogues installed

    Returns:
        Tuple[Table,Catalogue]: The loaded table and Catalogue structure describing the columns
    """
    catalogue = get_known_catalogue_info(name=name)
    catalogue_path = reference_catalogue_directory / catalogue.file_name
    table = Table.read(catalogue_path)

    if name == "SUMSS":
        table[catalogue.flux_col] = table[catalogue.flux_col] * u.mJy
    if name == "ICRF":
        return table, catalogue

    table[catalogue.flux_col] = table[catalogue.flux_col].to(u.Jy).value

    return table, catalogue


def get_rms_image_info(rms_path: Path) -> RMSImageInfo:
    """Extract information about the RMS image and construct a representative stucture

    Args:
        rms_path (Path): The RMS image that will be presented

    Returns:
        RMSImageInfo: Extracted RMS image information
    """

    with fits.open(rms_path) as rms_image:
        rms_header = rms_image[0].header
        rms_data = rms_image[0].data

    valid_pixels = np.sum(np.isfinite(rms_data) & (rms_data != 0))

    ra_cell_size = rms_header["CDELT1"]
    dec_cell_size = rms_header["CDELT2"]
    pixel_area = np.abs(ra_cell_size * dec_cell_size)

    area = pixel_area * valid_pixels

    rms_image_info = RMSImageInfo(
        path=rms_path,
        header=rms_header,
        shape=rms_data.shape,
        no_valid_pixels=valid_pixels,
        area=area,
    )

    return rms_image_info


def make_validator_axes_layout(fig: Figure, rms_path: Path) -> ValidatorLayout:
    """Create the figure layout to use for the quick look validation plot.

    Args:
        fig (Figure): The figure canvas to add the axes to
        rms_path (Path): Path to the RMS image that will be presented. Loaded to access the WCS

    Returns:
        ValidatorLayout: Representation of all axes objects
    """
    # My friendship with gridspec is over, now subplot_mosaic is my best friend
    # Using the following encoding:
    # T = text / field info
    # F = flux / source counts
    # A = astrometry
    # P = PSF
    # B = BANE noise map
    # S = SUMMS flux comparison
    # s = SUMMS astrometry comparison
    # N = NVSS flux comparison
    # n = NVSS astrometry comparison
    rms_wcs = WCS(fits.getheader(rms_path)).celestial
    ax_dict = fig.subplot_mosaic(
        """
        TFPA
        BBSs
        BBNn
        """,
        per_subplot_kw={
            "B": {"projection": rms_wcs},
        },
        subplot_kw={
            "aspect": "equal",
        },
    )
    # Remove the axes that are not used
    # TODO: Actually turn this back on with information
    _ = ax_dict["T"].axis("off")
    # Set the axes that are shared
    _ = ax_dict["N"].sharex(ax_dict["S"])
    _ = ax_dict["n"].sharex(ax_dict["s"])

    validator_layout = ValidatorLayout(
        ax_rms=ax_dict["B"],
        ax_legend=ax_dict["T"],
        ax_psf=ax_dict["P"],
        ax_counts=ax_dict["F"],
        ax_brightness1=ax_dict["N"],
        ax_brightness2=ax_dict["S"],
        ax_astrometry=ax_dict["A"],
        ax_astrometry1=ax_dict["n"],
        ax_astrometry2=ax_dict["s"],
    )

    return validator_layout


def plot_rms_map(fig: Figure, ax: plt.Axes, rms_path: Path) -> plt.Axes:
    """Add the RMS image to the figure

    Args:
        fig (Figure): Figure that contains the axes object
        ax (plt.Axes): The axes that will be plotted
        rms_path (Path): Location of the RMS image

    Returns:
        plt.Axes: The axes object with the plotted RMS image
    """

    rms_data = fits.getdata(rms_path)

    # Convert Jy/beam to uJy/beam
    im = ax.imshow(
        np.log10(rms_data * 1e6), vmin=2.0, vmax=3.0, origin="lower", cmap="YlOrRd"
    )

    ax.grid(color="0.5", ls="solid")
    ax.set_xlabel("RA (J2000)")
    ax.set_ylabel("Dec (J2000)")

    cbar = fig.colorbar(
        im,
        cax=None,
        ax=ax,
        use_gridspec=True,
        pad=0.09,
        shrink=0.5,
        orientation="vertical",
    )
    cbar.set_label(r"Image rms ($\mu$Jy/beam)", fontsize=F_SMALL)
    yt = np.log10([100, 200, 300, 500, 1000])
    ytl = [f"{10. ** lx:.0f}" for lx in yt]
    cbar.set_ticks(yt)
    cbar.set_ticklabels(ytl)

    gal_col = "b"
    overlay = ax.get_coords_overlay("galactic")
    overlay.grid(color=gal_col, ls="dashed", alpha=0.6)
    overlay[0].tick_params(colors=gal_col)
    overlay[1].tick_params(colors=gal_col)
    overlay[0].set_axislabel("Galactic long.", color=gal_col)
    overlay[1].set_axislabel("Galactic lat.", color=gal_col)

    ax.set_title("Image noise", loc="left", fontsize=F_MED)

    return ax


def get_source_counts(
    fluxes: np.ndarray,
    area: float,
    minlogf: float = -4,
    maxlogf: float = 2,
    Nbins: int = 81,
) -> SourceCounts:
    """Derive source counts for a set of fluxes and known area

    Args:
        fluxes (np.ndarray): The fluxes in Jy to count
        area (float): Area over which the sources were collected
        minlogf (float, optional): The minimum bin edge, in Jy. Defaults to -4.
        maxlogf (float, optional): The maximum bin edgem, in Jy. Defaults to 2.
        Nbins (int, optional): Number of bins to include in the source counts. Defaults to 81.

    Returns:
        SourceCounts: Source counts and their properties
    """
    logger.info(
        f"Computing source counts for {len(fluxes)} sources, {Nbins=}, {area=:.2f} sq.deg."
    )

    logf = np.linspace(minlogf, maxlogf, Nbins)
    f = np.power(10.0, logf)

    area_sky_total = 360.0 * 360.0 / np.pi
    solid_angle = 4 * np.pi * area / area_sky_total

    counts, bins = np.histogram(fluxes, f)
    binsmid = 0.5 * ((bins[:-1]) + (bins[1:]))
    ds = bins[1:] - bins[:-1]

    counts_err = np.array(np.sqrt(counts), dtype=int)

    scount = np.power(binsmid, 2.5) * counts / (ds * solid_angle)
    scount_err = np.power(binsmid, 2.5) * 1.0 * counts_err / (ds * solid_angle)

    source_counts = SourceCounts(
        bins=bins,
        bin_center=0.5 * (bins[1:] + bins[:-1]),
        counts_per_bin=counts,
        counts_per_bin_err=counts_err,
        euclid_counts=scount,
        euclid_counts_err=scount_err,
    )

    return source_counts


def plot_source_counts(
    catalogue: Table,
    rms_info: RMSImageInfo,
    ax: plt.Axes,
    freq: Optional[float] = None,
    dezotti: Optional[Table] = None,
    skads: Optional[Table] = None,
) -> plt.Axes:
    """Create a figure of source counts from a astropy Table. If
    `freq` and either `dezotti` / `skads` are supplied then these
    precomputed source count tables are also included in the
    panel.

    When computing the source counts for `catalogue`, only a
    minimumal set of corrections are derived and applied.

    Args:
        catalogue (Table): The catalogue to derive source counts for
        rms_info (RMSImageInfo): Look up information from the RMS file that catalogue was constructed against
        ax (plt.Axes): The axes panel the counts will be plottedd on
        freq (Optional[float], optional): Frequency that the source catalogue. Used to scale the Dezotti and SKADS tables. Defaults to None.
        dezotti (Optional[Table], optional): Loaded reference table of Dezotti source counts. Defaults to None.
        skads (Optional[Table], optional): Loaded reference table of SKADS source counts. Defaults to None.

    Returns:
        plt.Axes: The axes object used for plotting
    """
    # TODO: Is the freq column needed anymore

    fluxes = catalogue["int_flux"]

    source_counts = get_source_counts(fluxes=fluxes, area=rms_info.area)

    ax.errorbar(
        source_counts.bin_center * 1e3,
        source_counts.euclid_counts,
        yerr=source_counts.euclid_counts_err,
        fmt=".",
        color="darkred",
        label=f"Raw Component Catalogue - {len(fluxes)} sources",
    )

    if dezotti is not None and freq is not None:
        spectral_index_scale = (freq / 1.4e9) ** -0.8
        ax.errorbar(
            np.power(10.0, dezotti["col1"]) * 1.0e3 * spectral_index_scale,
            dezotti["col2"] * np.power(spectral_index_scale, 1.5),
            yerr=[
                dezotti["col4"] * np.power(spectral_index_scale, 1.5),
                dezotti["col3"] * np.power(spectral_index_scale, 1.5),
            ],
            fmt="d",
            color="grey",
            alpha=0.2,
            label="de Zotti et al. 2010",
        )

    if skads is not None and freq is not None:
        spectral_index_scale = (freq / 1.4e9) ** -0.8
        ax.errorbar(
            skads["fl_bin_mid"] * 1.0e3 * spectral_index_scale,
            skads["SC"] * np.power(spectral_index_scale, 1.5),
            yerr=skads["SC_err"] * np.power(spectral_index_scale, 1.5),
            fmt=":",
            color="k",
            label="Wilman et al. 2008",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Flux Density (mJy)")
    ax.set_ylabel(r"$\frac{dN}{dS} S^{\frac{5}{2}}$ (Jy$^{\frac{3}{2}}$ sr$^{-1}$)")
    ax.set_xlim(0.2, 1.0e4)
    ax.set_ylim(5.0e-2, 1.0e4)
    ax.grid()
    ax.legend(loc="lower right", fontsize=F_SMALL)

    return ax


def match_nearest_neighbour(
    table1: Table,
    table2: Table,
    catalogue1: Catalogue,
    catalogue2: Catalogue,
    radius: float = 10,
) -> MatchResult:
    """Match two catalogues together, and construct common properties.

    Args:
        table1 (Table): The catalogue table from survey one
        table2 (Table): The catalogue table from survey two
        catalogue1 (Catalogue): Catalogue metadata for survey one
        catalogue2 (Catalogue): Catalogue metadata for survey two
        radius (float, optional): Maximum matching radius. Defaults to 10.

    Returns:
        MatchResult: Object containing source matches and common properties
    """

    pos1 = SkyCoord(
        table1[catalogue1.ra_col], table1[catalogue1.dec_col], unit="deg,deg"
    )
    pos2 = SkyCoord(
        table2[catalogue2.ra_col], table2[catalogue2.dec_col], unit="deg,deg"
    )

    idx, sep, _ = pos1.match_to_catalog_sky(catalogcoord=pos2, nthneighbor=1)
    mask = sep < (radius * u.arcsecond)

    idx = idx[mask]

    match_result = MatchResult(
        name1=catalogue1.survey,
        name2=catalogue2.survey,
        pos1=pos1[mask],
        pos2=pos2[idx],
        freq1=catalogue1.freq,
        freq2=catalogue2.freq,
        flux1=table1[catalogue1.flux_col].value[mask]
        if catalogue1.flux_col != "None"
        else None,
        flux2=table2[catalogue2.flux_col].value[idx]
        if catalogue2.flux_col != "None"
        else None,
    )

    return match_result


def plot_astrometry_comparison(
    fig: Figure, ax: plt.Axes, match_result: MatchResult
) -> plt.Axes:
    """Plot the astrometry of cross matches from a match result set

    Args:
        fig (Figure): The figure canvas plotting on
        ax (plt.Axes): The Axes being plotted on
        match_result (MatchResult): The set of sources cross-matched and found in common

    Returns:
        plt.Axes: The Axes plotted on
    """
    logger.info(
        f"Plotting astrometry match between {match_result.name1} and {match_result.name2}"
    )
    if len(match_result.pos1) == 0:
        ax.set_xlim(-8.0, 8.0)
        ax.set_ylim(-8.0, 8.0)
        ax.text(0.0, 0.0, f"No data for {match_result.name2}", va="center", ha="center")

        return ax

    pos1, pos2 = match_result.pos1, match_result.pos2

    # Get the sepatations between two points, and their angle between them
    seps = pos1.separation(pos2)
    pas = pos1.position_angle(pos2)

    # Apply corrections (were in the original code)
    # TODO: Find out why this is necessary. Sky is curved nonsense doesn't depend on angle between sources
    err_x = (seps * -np.sin(pas.radian)).to(u.arcsec).value
    err_y = (seps * np.cos(pas.radian)).to(u.arcsec).value

    mean_x, mean_y = err_x.mean(), err_y.mean()
    std_x, std_y = err_x.std(), err_y.std()

    ax.plot(err_x, err_y, ".r", zorder=0, ms=2)
    ax.errorbar(mean_x, mean_y, xerr=std_x, yerr=std_y, fmt="ok")

    ax.grid()
    ax.axvline(0, color="black", ls=":")
    ax.axhline(0, color="black", ls=":")

    ax.text(
        -7.0,
        7.0,
        match_result.name2,
        fontsize=F_MED,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
    )
    ax.text(
        -7.0,
        7.0,
        f"{match_result.name2} - {len(err_x)} matches",
        fontsize=F_MED,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
    )

    ax.set(
        xlim=(-8, 8), ylim=(-8, 8), xlabel="Offset (arcsec)", ylabel="Offset (arcsec)"
    )
    le_a1 = r" $\epsilon_{SU} : ({%.1f}\pm{%.1f},{%.1f}\pm{%.1f})$" % (
        mean_x,
        std_x,
        mean_y,
        std_y,
    )
    ax.text(
        7.6,
        7.6,
        le_a1,
        va="top",
        ha="right",
        fontsize=F_MED,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
    )
    return ax


def plot_flux_comparison(
    fig: Figure, ax: plt.Axes, match_result: MatchResult
) -> plt.Axes:
    """Create a flux comparison plot showing the flux densities from two catalogues compared
    to one another.

    Args:
        fig (Figure): The figure canvas that the axes is on
        ax (plt.Axes): The axes object that will be render the plot
        match_result (MatchResult): A result set of the cross-match between two catalogues

    Returns:
        plt.Axes: The aces object that was used for plotting
    """
    if len(match_result.pos1) == 0:
        ax.loglog([2.0, 2000.0], [2.0, 2000.0], "ow")
        ax.text(
            100.0, 100.0, f"No data for {match_result.name2}", va="center", ha="center"
        )

        return ax

    flux1 = match_result.flux1
    flux2 = match_result.flux2

    one2one = np.array([0.002, 8])
    spectral_index_scale = (match_result.freq2 / match_result.freq1) ** -0.8

    ax.loglog(flux1, flux2, "ok", ms=2)
    ax.loglog(one2one, one2one, "-", c="r")
    ax.loglog(one2one, one2one * spectral_index_scale, "--", c="r")
    ax.set(
        ylabel=f"{match_result.name2} Integrated Flux (Jy)",
        xlabel=f"{match_result.name1} Integrated Flux (Jy)",
    )

    ax.grid()

    return ax


def plot_psf(fig: Figure, ax: plt.Axes, rms_info: RMSImageInfo) -> plt.Axes:
    """Create a plot highlighting the synthesised beam recorded in the
    RMS image header

    Args:
        fig (Figure): Fogire canvas being used
        ax (plt.Axes): The axes object that will be used for plotting
        rms_info (RMSImageInfo): Extracted information from the RMS image

    Returns:
        plt.Axes: The aces object used for plotting
    """

    bmaj = rms_info.header["BMAJ"] * 3600
    bmin = rms_info.header["BMIN"] * 3600

    ax.plot(bmin, bmaj, "ok", ms=3)
    ax.set(
        xlim=(5.0, 20.0),
        ylim=(5.0, 40.0),
        xlabel="PSF minor axis (arcsec)",
        ylabel="PSF major axis (arcsec)",
    )

    ax.grid()

    return ax


def create_validation_plot(
    rms_image_path: Path,
    source_catalogue_path: Path,
    output_path: Path,
    reference_catalogue_directory: Path,
) -> Path:
    """Create a simple multi-panel validation figure intended to asses
    the correctness of an image and associated source catalogue.

    The image described by `rms_image_path` should be a FITS file. The
    WCS of this file is used for plotting and rreading the synthesised
    beam information using the standard CRVAL/BMAJ/BMIN keywords.

    The source catalogue is read using astropy.table.Table. This routine
    also expects that some level of units are embedded in the catalogue.
    For Aegean produced catalogues this is the case.

    The reference_catalogue_path sets the directory to look into when
    searching for the reference ICRF, NVSS and SUMSS cataloues.

    Args:
        rms_image_path (Path): The RMS fits image the source catalogue was constructed against.
        source_catalogue_path (Path): The source catalogue.
        output_path (Path): The output path of the figure to create
        reference_catalogue_directory (Path): The directory that contains the reference ICRF, NVSS and SUMSS catalogues.

    Returns:
        Path: The output path of the figure
    """
    rms_info = get_rms_image_info(rms_path=rms_image_path)

    dezotti_path = Path(
        pkg_resources.resource_filename("flint", "data/source_counts/de_zotti_1p4.txt")
    )
    skads_path = Path(
        pkg_resources.resource_filename("flint", "data/source_counts/SKADS_1p4GHz.fits")
    )

    logger.info(f"Loading {dezotti_path}")
    dezotti = Table.read(dezotti_path, format="ascii")
    logger.info(f"Loading {skads_path=}")
    skads = Table.read(skads_path)

    height = 8.0
    width = height * np.sqrt(2.0)

    fig = plt.figure(figsize=(width, height))

    validator_layout = make_validator_axes_layout(fig=fig, rms_path=rms_image_path)

    logger.info(f"Loading {source_catalogue_path=}")
    askap_table = Table.read(source_catalogue_path)
    askap_cata = Catalogue(
        survey="ASKAP",
        file_name=source_catalogue_path.name,
        freq=rms_info.header["CRVAL3"],
        ra_col="ra",
        dec_col="dec",
        name_col="source",
        flux_col="int_flux",
        maj_col="a",
        min_col="b",
        pa_col="pa",
    )
    plot_source_counts(
        catalogue=askap_table,
        rms_info=rms_info,
        ax=validator_layout.ax_counts,
        freq=rms_info.header["CRVAL3"],
        dezotti=dezotti,
        skads=skads,
    )

    plot_rms_map(fig=fig, ax=validator_layout.ax_rms, rms_path=rms_info.path)

    plot_psf(fig=fig, ax=validator_layout.ax_psf, rms_info=rms_info)

    ierf_table, ierf_catalogue = load_known_catalogue(
        name="ICRF", reference_catalogue_directory=reference_catalogue_directory
    )
    ierf_match = match_nearest_neighbour(
        table1=askap_table,
        table2=ierf_table,
        catalogue1=askap_cata,
        catalogue2=ierf_catalogue,
    )
    plot_astrometry_comparison(
        fig=fig, ax=validator_layout.ax_astrometry, match_result=ierf_match
    )

    sumss_table, sumss_catalogue = load_known_catalogue(
        name="SUMSS", reference_catalogue_directory=reference_catalogue_directory
    )
    sumss_match = match_nearest_neighbour(
        table1=askap_table,
        table2=sumss_table,
        catalogue1=askap_cata,
        catalogue2=sumss_catalogue,
    )

    plot_astrometry_comparison(
        fig=fig, ax=validator_layout.ax_astrometry1, match_result=sumss_match
    )
    plot_flux_comparison(
        fig=fig, ax=validator_layout.ax_brightness1, match_result=sumss_match
    )

    nvss_table, nvss_catalogue = load_known_catalogue(
        name="NVSS", reference_catalogue_directory=reference_catalogue_directory
    )
    nvss_match = match_nearest_neighbour(
        table1=askap_table,
        table2=nvss_table,
        catalogue1=askap_cata,
        catalogue2=nvss_catalogue,
    )

    plot_astrometry_comparison(
        fig=fig, ax=validator_layout.ax_astrometry2, match_result=nvss_match
    )
    plot_flux_comparison(
        fig=fig, ax=validator_layout.ax_brightness2, match_result=nvss_match
    )

    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")

    return output_path


def get_parser() -> ArgumentParser:
    """Create the argument parser for the validation plot creation

    Returns:
        ArgumentParser: CLI entry point
    """
    parser = ArgumentParser(
        description="Create a validation figure to highlight the reliability of some process continuum field"
    )

    parser.add_argument(
        "rms_image_path", type=Path, help="Path to the RMS image of the field"
    )
    parser.add_argument(
        "source_catalogue_path",
        type=Path,
        help="Path to the source catalogue. At present on Aegean formatted component catalogues supported. ",
    )
    parser.add_argument(
        "output_path", type=Path, help="Location of the output figure to create. "
    )
    parser.add_argument(
        "--reference-catalogue-directory",
        type=Path,
        default=Path("."),
        help="Directory container the reference ICFS, NVSS and SUMSS catalogues. These are known catalogues and expect particular file names. ",
    )

    return parser


def cli() -> None:
    """CLI entry point for validation plot creation"""
    parser = get_parser()

    args = parser.parse_args()
    create_validation_plot(
        rms_image_path=args.rms_image_path,
        source_catalogue_path=args.source_catalogue_path,
        output_path=args.output_path,
        reference_catalogue_directory=args.reference_catalogue_directory,
    )


if __name__ == "__main__":
    cli()
