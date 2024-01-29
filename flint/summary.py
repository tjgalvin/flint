"""Summary containers intended to hold general information obtained throughout a processing run
"""
from __future__ import (  # Used for mypy/pylance to like the return type of MS.with_options
    annotations,
)

from pathlib import Path
from typing import NamedTuple, Optional

from flint.logging import logger


class FieldSummary(NamedTuple):
    """The main information about a processed field.

    There are known issues around serialising astropy units within a dask/prefect environment,
    for example: https://github.com/astropy/astropy/issues/11317, For this reason these
    attributes will mostly be primative types, or structures that can
    be trusted to be serialisable.
    """

    sbid: str
    """SBID of processed field"""
    cal_sbid: str
    """SBID of the bandpass calibrator"""
    radec_j2000_str: str
    """The RA and Dec (J2000) of the field centre as a string"""
    gal_lb_str: str
    """The Galactic coordinates of the field centre as a string"""
    start_time_str: str = ""
    """The start time of the observation"""
    integration_time_str: str = ""
    """The amount of integration time of the observation (seconds)"""
    hour_angle_range_str: str = ""
    """Range in the hour angles of the observation"""
    median_rms_ujy: Optional[float] = None
    """The median rms of the field in uJy"""
    no_components: Optional[int] = None
    """Number of components found from the source finder"""
    holography_path: Optional[Path] = None
    """Path to the file used for holography"""

    def with_options(self, **kwwargs) -> FieldSummary:
        pass


def create_field_summary() -> FieldSummary:
    pass


def update_field_summary() -> FieldSummary:
    pass
