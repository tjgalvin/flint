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

from casacore.tables import table
from astropy.table import Table
import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np

from flint.logging import logger
from flint.ms import MS, get_phase_dir_from_ms, get_freqs_from_ms
from flint.sky_model import generate_pb
from flint.utils import get_packaged_resource_path


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


def find_sources_to_peel(
    ms: MS, field_idx: int = 0, max_sep_deg: float = 5, cutoff: float = 0.1
) -> Table:
    """Obtain a set of sources to peel from a reference candidate set. This will
    evaluate whether a source should be peels based on two criteria:

    - if it is below a nominal primary beam cut off level (10 percent, assuming a gaussian primary beam)
    - if the sources is within some separation of the imaging center

    Args:
        ms (MS): The measurement set that is being considered
        field_idx (int, optional): Which field in the MS to draw the position from. Defaults to 0.
        max_sep_deg (float, optional): The radius from the imaging center a source needs to be (in degrees) for it to be considered a source to peel. Defaults to 5.
        cutoff (float, optional): The primary beam attentuation level a source needs to be below for it to be considered a source to peel. Defaults to 0.1.

    Returns:
        Table: Collection of sources to peel from the reference table. Column names are Name, RA, Dec, Aperture. This is the package table
    """
    max_sep = max_sep_deg * u.deg

    logger.debug(f"Extracting image direction for {field_idx=}")
    image_coord = get_phase_dir_from_ms(ms=ms)

    logger.info(f"Considering sources to peel around {image_coord=}")

    peel_srcs_tab = load_known_peel_sources()

    freqs = get_freqs_from_ms(ms=ms)
    nominal_freq = np.mean(freqs)
    logger.info(f"The nominal frequency is {nominal_freq / 1e6}MHz")

    peel_srcs = []

    # TODO: Make this a mapping function
    for src in peel_srcs_tab:
        src_coord = SkyCoord(src["RA"], src["Dec"], unit=(u.hourrangle, u.degree))
        offset = image_coord.separation(src_coord)

        if offset > max_sep:
            logger.debug(
                f"{src['Name']} offset is {offset}, max separation is {max_sep}"
            )
            continue

        taper = generate_pb(
            pb_model="airy", freqs=nominal_freq, aperture=12 * u.m, offset=offset
        )
        if taper.atten > cutoff:
            logger.info(
                f"{src['Name']} attenuation {taper.atten} is above {cutoff=} (in field of view)"
            )
            continue

        peel_srcs.append(src)

    peel_tab = Table(peel_srcs)

    return peel_tab


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
        tab = find_sources_to_peel(ms=ms)
        logger.info("Sources to peel")
        logger.info(tab)

    if args.mode == "peel":
        ms = MS(path=args.ms, column=args.data_column)

        potato_peel(ms=ms, potato_peel=args.potato_container)


if __name__ == "__main__":
    cli()
