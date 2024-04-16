"""Utilities the connect to Stefan Duchesne potatopeel module, which is responsible for
peeling sources in racs. The repository is available at:

https://gitlab.com/Sunmish/potato/-/tree/main

Potato stands for "Peel Out That Annoying Terrible Object". 

Although this is a python module, for the moment it is expected
to be in a singularity container. There are several reasons 
for this, but the principal one is that the numba module used
by potatopeel may be difficult to get working correctly alongside
dask and flint. Keeping it simple at this point is the main aim. 

The CLI of potatopeel is called across two stages:

configFile="${source_name}_peel.cfg"
$container peel_configuration.py "${configFile}" \
    --image_size=${IMSIZE} \
    --image_scale=${scale} \
    --image_briggs=${PEEL_BRIGGS} \
    --image_channels=4 \
    --image_minuvl=0 \
    --peel_size=1000 \
    --peel_scale=${scale} \
    --peel_channels=${chansOut} \
    --peel_nmiter=7 \
    --peel_minuvl=0 \
    --peel_multiscale

$container potato ${TMP_DIR}${obsid}.ms ${source_ra} ${source_dec} ${peel_fov} ${subtract_rad} \
    --config ${configFile} \
    -solint ${PEEL_SOLINT} \
    -calmode ${PEEL_CALMODE}  \
    -minpeelflux ${threshold1} \
    --name ${source_name} \
    --refant 1 \
    --direct_subtract \
    --no_time \
    --intermediate_peels \
    --tmp ${TMP_DIR}

"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Union

from casacore.tables import table
from astropy.table import Table
import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np

from flint.imager.wsclean import WSCleanOptions
from flint.logging import logger
from flint.ms import MS, get_phase_dir_from_ms, get_freqs_from_ms
from flint.sky_model import generate_pb
from flint.utils import get_packaged_resource_path, generate_strict_stub_wcs_header


def load_known_peel_sources() -> Table:
    """Locate and load the packaged set of known sources to peel. These sources
    are drawn from Duchesne et. al. (2023), Table 3.

    Returns:
        Table: The astropy Table of candidate sources to remove.
    """
    peel_srcs_csv = get_packaged_resource_path("flint", "data/peel/known_sources.csv")

    logger.info(f"Loading in {peel_srcs_csv=}")
    peel_tab = Table.read(peel_srcs_csv)

    logger.info(f"Loaded {len(peel_tab)} sources. ")

    return peel_tab


def source_within_image_fov(
    source_coord: SkyCoord,
    beam_coord: SkyCoord,
    image_size: int,
    pixel_scale: Union[u.Quantity, str],
) -> bool:
    """Evaluate whether a source will be within the field of view
    of an image.

    Args:
        source_coord (SkyCoord): The source position to consider
        beam_coord (SkyCoord): The center beam position to consider
        image_size (int): The image size. This assumes a square image.
        pixel_scale (Union[u.Quantity, str]): The pixel size, assuming square pixels.

    Returns:
        bool: Whether the source is expected to be in the field of view
    """
    # TODO: Allow the wcs to be passed in

    wcs = generate_strict_stub_wcs_header(
        position_at_image_center=beam_coord,
        image_shape=(image_size, image_size),
        pixel_scale=pixel_scale,
    )

    x, y = wcs.all_world2pix(source_coord.ra.deg, source_coord.dec.deg, 0)

    # logger.info(
    #     f"Source coordinate {(source_coord.ra.deg, source_coord.dec.deg ) } for {beam_coord} is at pixel {(x, y)}. "
    # )

    # Since the reference pixel is at the image center, then the valid domain
    # is between 0 and image size
    source_in_image = 0 < x <= image_size and 0 < y <= image_size

    # logger.info(f"The {source_in_image=}")
    return source_in_image


def find_sources_to_peel(
    ms: MS,
    image_options: WSCleanOptions,
    field_idx: int = 0,
    maximum_offset: float = 30,
    minimum_apparent_brightness: float = 0.5,
) -> Union[Table, None]:
    """Obtain a set of sources to peel from a reference candidate set. This will
    evaluate whether a source should be peels based on two criteria:

    - if it is below a nominal primary beam cut off level (10 percent, assuming a gaussian primary beam)
    - if the sources is within some separation of the imaging center

    Args:
        ms (MS): The measurement set that is being considered
        image_options (WSCleanOptions): The imaging parameters that will be used to compute a placeholder WCS
        field_idx (int, optional): Which field in the MS to draw the position from. Defaults to 0.
        maximum_offset (float, optional): The largest separation, in degrees, before a source is ignored. Defaults to 30.0.
        minimum_apparent_brightness (float, optional): The minimum apparent brightnessm, in Jy, a source should be before attempting to peel. Defaults to 0.5.

    Returns:
        Union[Table,None]: Collection of sources to peel from the reference table. Column names are Name, RA, Dec, Aperture. This is the package table. If no sources need to be peeled None is returned.
    """
    image_size, pixel_scale = None, None

    if isinstance(image_options, WSCleanOptions):
        image_size = image_options.size
        pixel_scale = image_options.scale
    else:
        raise TypeError(f"{type(image_options)=} is not known. ")

    logger.debug(f"Extracting image direction for {field_idx=}")
    image_coord = get_phase_dir_from_ms(ms=ms)

    logger.info(f"Considering sources to peel around {image_coord=}")

    peel_srcs_tab = load_known_peel_sources()

    freqs = get_freqs_from_ms(ms=ms)
    nominal_freq = np.mean(freqs) * u.Hz
    logger.info(f"The nominal frequency is {nominal_freq / 1e6}MHz")

    peel_srcs = []

    # TODO: Make this a mapping function
    for src in peel_srcs_tab:
        src_coord = SkyCoord(src["RA"], src["Dec"], unit=(u.hourangle, u.degree))
        offset = image_coord.separation(src_coord)

        if source_within_image_fov(
            source_coord=src_coord,
            beam_coord=image_coord,
            image_size=image_size,
            pixel_scale=pixel_scale,
        ):
            logger.debug(
                f"Source {src['name']} within image field of view of image with {image_size}x{image_size} and {pixel_scale=}"
            )
            continue

        taper = generate_pb(
            pb_type="airy", freqs=nominal_freq, aperture=12 * u.m, offset=offset
        )
        assert isinstance(taper.atten, float), "Expected a float"

        if offset > maximum_offset * u.deg:
            continue

        # if taper.atten[0] * tab[] > cutoff:
        #     logger.info(
        #         f"{src['Name']} attenuation {taper.atten} is above {cutoff=} (in field of view)"
        #     )
        #     continue

        peel_srcs.append(src)

    return Table(rows=peel_srcs) if peel_srcs else None


def prepare_ms_for_potato(ms: MS) -> MS:
    """The potatopeel software requires the data column being operated against to be
    called DATA. This is a requirement of CASA and its gaincal / applysolution task.

    If there is already a DATA column and this is not the nominated data column described
    by the MS then it is removed, and the nominated data column is renamed.

    If the DATA column already exists and it is the nominated column described by
    MS then this is returned. If there is a CORRECTED_DATA column in this situation
    it is removed.

    Args:
        ms (MS): The measurement set that will be edited

    Raises:
        ValueError: Raised when a column name is expected but can not be found.

    Returns:
        MS: measurement set with the data column moved into the correct location
    """
    data_column = ms.column

    # If the data column already exists and is the nominated column, then we should
    # just return, ya scally-wag
    if data_column == "DATA":
        logger.info(f"{data_column=} is already DATA. No need to rename. ")

        with table(str(ms.path), readonly=False, ack=False) as tab:
            if "CORRECTED_DATA" in tab.colnames():
                logger.info(f"Removing CORRECTED_DATA column from {ms.path}")
                tab.removecols("CORRECTED_DATA")

        return ms

    with table(str(ms.path), ack=False, readonly=False) as tab:
        colnames = tab.colnames()
        if data_column not in colnames:
            raise ValueError(
                f"Column {data_column} not found in {str(ms.path)}. Columns found: {colnames}"
            )

        # In order to rename the data_column to DATA, we need to make sure that
        # there is not an existing DATA column
        if "DATA" in colnames:
            logger.info("Removing the existing DATA column")
            tab.removecols("DATA")

        tab.renamecol(data_column, "DATA")

        # Remove any CORRECT_DATA column, should it exist, as
        # potatopeel will create it
        if "CORRECTED_DATA" in colnames:
            logger.info(f"Removing CORRECTED_DATA column from {ms.path}")
            tab.removecols("CORRECTED_DATA")

    return ms.with_options(column="DATA")


def potato_peel(ms: MS, potato_container: Path) -> MS:
    """Peel out sources from a measurement set using PotatoPeel. Candidate sources
    from a known list of sources (see Table 3 or RACS-Mid paper) are considered.

    Args:
        ms (MS): The measurement set to peel out known sources
        potato_container (Path): Location of container with potatopeel software installed

    Returns:
        MS: Updated measurement set
    """
    potato_container = potato_container.absolute()

    logger.info(f"Will attempt to peel the {ms=}")
    logger.info(f"Using the potato peel container {potato_container}")

    peel_tab = find_sources_to_peel(ms=ms)

    if len(peel_tab) == 0:
        logger.info("No sources to peel. ")
        return ms

    ms = prepare_ms_for_potato(ms=ms)

    return ms


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="A simple interface into the potatopeel software"
    )

    subparser = parser.add_subparsers(dest="mode")

    check_parser = subparser.add_parser(
        "check", help="Check whether there are sources to peel from a measurement set"
    )

    check_parser.add_argument(
        "ms", type=Path, help="Path to the measurement set that will be considered"
    )
    check_parser.add_argument(
        "--image-size",
        type=int,
        default=8000,
        help="The number of pixels that make up a square image. Used to determine if a source is within FoV",
    )
    check_parser.add_argument(
        "--pixel-scale",
        type=str,
        default="2.5arcsec",
        help="The size of a pixel in an astropy-understood unit. Used to assess whether a source is within the image FoV. ",
    )

    list_parser = subparser.add_parser(  # noqa
        "list", help="List the known candidate sources available for peeling. "
    )

    peel_parser = subparser.add_parser(
        "peel", help="Attempt to peel sources using potatopeel"
    )
    peel_parser.add_argument(
        "ms", type=Path, help="Path the to the measurement set with a source to peel"
    )

    peel_parser.add_argument(
        "--potato-container",
        type=Path,
        default=Path("./potato.sif"),
        help="Path to the singularity container that represented potatopeel",
    )
    peel_parser.add_argument(
        "--data-column",
        type=str,
        default="DATA",
        help="The column name that contains data with a source that needs to be peeled",
    )

    return parser


def cli():
    parser = get_parser()

    args = parser.parse_args()

    if args.mode == "list":
        tab = load_known_peel_sources()
        logger.info("Known candidate sources to peel")
        logger.info(tab)

    if args.mode == "check":
        ms = MS(path=args.ms)
        image_options = WSCleanOptions(size=args.image_size, scale=args.pixel_scale)

        tab = find_sources_to_peel(ms=ms, image_options=image_options)
        logger.info("Sources to peel")
        logger.info(tab)

    if args.mode == "peel":
        ms = MS(path=args.ms, column=args.data_column)

        potato_peel(ms=ms, potato_peel=args.potato_container)


if __name__ == "__main__":
    cli()
