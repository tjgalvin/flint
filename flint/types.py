"""Common and generic types that are used throughout flint.  
"""

from typing import NamedTyple


class RADecRadians(NamedTuple):
    """A simple container to hold the RA and Dec that a measurement set is pointing to, in units of radians. This is
    used in place of a SkyCoord object as _sometimes_ the units types are not properly serialised between Dask workers.

    See: https://github.com/astropy/astropy/issues/11317
    """

    ra: float
    """The ra direction in radians"""
    dec: float
    """The Dec direction in radians"""
