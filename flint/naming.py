"""Attempts to centralise components to do with naming of pipeline files and data
products. 
"""
import re
from pathlib import Path
from typing import Union, Optional, Dict, NamedTuple

from flint.logging import logger


class RawNameComponents(NamedTuple):
    date: str
    """Date that the data were taken, of the form YYYY-MM-DD"""
    time: str
    """Time that the data were written"""
    beam: str
    """Beam number of the data"""
    spw: Optional[str] = None
    """If multiple MS were written as the data were in a high-frequency resolution mode, which segment"""


# Beginnings of some tests to write, ye old sea dog
# In [42]: e = re.compile("(?P<date>[0-9]{4}-[0-9]{1,2}-[0-9]{1,2})_(?P<time>[0-9]+)_(?P<beam>[0-9]+)(_(?P<spw>[0-9]+))*")
# In [43]: results = e.match(a)
# In [44]: results.groupdict()
# Out[44]: {'date': '2022-04-14', 'time': '100122', 'beam': '1', 'spw': None}
# In [45]: results = e.match(b)
# In [46]: results.groupdict()
# Out[46]: {'date': '2022-04-14', 'time': '100122', 'beam': '1', 'spw': '2'}
# In [47]: a
# Out[47]: '2022-04-14_100122_1'
# In [48]: b
# Out[48]: '2022-04-14_100122_1_2'


def raw_ms_format(in_name: str) -> Union[None, RawNameComponents]:
    """The typical ASKAP measurement written to the ingest disks
    has the form:

    >>> 2022-04-14_100122_1.ms

    and in the case of a multiple beams written out (in high frequency resolution mode)

    >>> 2022-04-14_100122_1_1.ms

    This function will attempt to break it up into its main parts
    and return the mapping.

    Args:
        in_name (str): The name of a file, presumably a measurement set. The left-most part will be examined for to identify the raw ASKAP naming scheme.

    Returns:
        Union[None,Dict[str,str]]: None if the raw ASKAP measurement set naming scheme was not detected, otherwise a dictionary representing its parts.
    """

    logger.debug(f"Matching {in_name}")
    regex = re.compile(
        "^(?P<date>[0-9]{4}-[0-9]{1,2}-[0-9]{1,2})_(?P<time>[0-9]+)_(?P<beam>[0-9]+)(_(?P<spw>[0-9]+))*"
    )
    results = regex.match(in_name)

    if results is None:
        logger.info(f"No results to {in_name} found")
        return None

    groups = results.groupdict()

    logger.info(f"Matched groups are: {groups}")

    return RawNameComponents(
        date=groups["date"], time=groups["time"], beam=groups["beam"], spw=groups["spw"]
    )


def extract_beam_from_name(name: Union[str, Path]) -> int:
    """Attempts to extract the beam number of a file name, presumably a measurement set,
    that has the form similar to the typical format of raw measurement sets freshly written
    to disk. Such a search might be useful when considering images where the beam information
    might not be attached in the FITS file (or otherwise passed around).

    Args:
        name (Union[str,Path]): The name to examine to search for the beam number.

    Raises:
        ValueError: Raised if the name was not was not successfully matched against the known format

    Returns:
        int: Beam number that extracted from the input name
    """

    name = str(name.name) if isinstance(name, Path) else name
    results = raw_ms_format(in_name=name)

    if results is None:
        raise ValueError(f"Unrecognised file name format for {name=}. ")

    return int(results.beam)


class AegeanNames(NamedTuple):
    """Base names that would be used in various Aegean related tasks"""

    bkg_image: Path
    rms_image: Path
    comp_cat: Path
    ds9_region: Path
    resid_image: Path


def create_aegean_names(base_output: str) -> AegeanNames:
    """Create the expected names for aegean and its tools.

    Args:
        base_output (str): The base name that aegean outputs are built from.

    Returns:
        AegeanNames: A collection of names to be produced by Aegean related tasks
    """
    return AegeanNames(
        bkg_image=Path(f"{base_output}_bkg.fits"),
        rms_image=Path(f"{base_output}_rms.fits"),
        comp_cat=Path(f"{base_output}_comp.fits"),
        ds9_region=Path(f"{base_output}_overlay.reg"),
        resid_image=Path(f"{base_output}_residual.fits"),
    )
