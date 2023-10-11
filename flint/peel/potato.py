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
from astropy.table import table
import astropy.units as u
from astropy.coordinates import SkyCoord

from flint.logging import logger
from flint.ms import MS, get_radec_direction
from flint.types import RADecRadians


def find_sources_to_peel(ms: MS):
    field_idx = 0
    logger.debug(f"Extracting image direction for {field_idx=}")
    image_direction: RADecRadians = get_radec_direction(ms=ms, field_idx=field_idx)
    image_coord = SkyCoord(image_direction.ra * u.rad, image_direction.dec * u.rad)

    logger.info(f"Considering sources to peel around {image_coord=}")

    # TODO: Get the known sources to peel


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
    potato_container = potato_container.absolute()

    logger.info(f"Will attempt to peel the {ms=}")
    logger.info(f"Using the potato peel container {potato_container}")

    ms = prepare_ms_for_potato(ms=ms)

    return ms


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="A simple interface into the potatopeel software"
    )

    parser.add_argument(
        "ms", type=Path, help="Path the to the measurement set with a source to peel"
    )

    parser.add_argument(
        "--potato-container",
        type=Path,
        default=Path("./potato.sif"),
        help="Path to the singularity container that represented potatopeel",
    )
    parser.add_argument(
        "--data-column",
        type=str,
        default="DATA",
        help="The column name that contains data with a source that needs to be peeled",
    )

    return parser


def cli():
    parser = get_parser()

    args = parser.parse_args()

    ms = MS(path=args.ms, column=args.data_column)

    potato_peel(ms=ms, potato_peel=args.potato_container)


if __name__ == "__main__":
    cli()
