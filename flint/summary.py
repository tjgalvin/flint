"""Summary containers intended to hold general information obtained throughout a processing run
"""
from __future__ import (  # Used for mypy/pylance to like the return type of MS.with_options
    annotations,
)

from pathlib import Path
from typing import NamedTuple, Optional, Union

import numpy as np
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, Longitude, Latitude
from astropy.table import Table
from astropy.time import Time

from flint.logging import logger
from flint.ms import MS, get_telescope_location_from_ms, get_times_from_ms
from flint.naming import get_sbid_from_path
from flint.source_finding.aegean import AegeanOutputs
from flint.validation import get_rms_image_info


class FieldSummary(NamedTuple):
    """The main information about a processed field. This structure is
    intended to store critical components that might be accumulated throughout
    processing of a pipeline, and may be most useful when data-products
    are zipped (or removed) throughout.

    There are known issues around serialising astropy units within a dask/prefect environment,
    for example: https://github.com/astropy/astropy/issues/11317, Ye be warned.
    """

    sbid: str
    """SBID of processed field"""
    cal_sbid: str
    """SBID of the bandpass calibrator"""
    centre: Optional[SkyCoord] = None
    """Centre of the field, which is calculated from the centre of an image"""
    integration_time: Optional[int] = None
    """The integration time of the observation (seconds)"""
    median_rms_ujy: Optional[float] = None
    """The median rms of the field in uJy"""
    no_components: Optional[int] = None
    """Number of components found from the source finder"""
    holography_path: Optional[Path] = None
    """Path to the file used for holography"""
    selfcal_round: Optional[int] = None
    """The self-cal round"""
    location: Optional[EarthLocation] = None
    """The location of the telescope stored as (X,Y,Z) in meters"""
    ms_times: Optional[Time] = None
    """The unique scan times of integrations stored in the measurement set"""
    hour_angles: Optional[Longitude] = None
    """Computed hour-angles of the field"""
    elevations: Optional[Latitude] = None
    """Computed elevations of the field"""
    median_rms: Optional[float] = None
    """The meanian RMS computed from an RMS image"""

    def with_options(self, **kwargs) -> FieldSummary:
        prop = self._asdict()
        prop.update(**kwargs)

        return FieldSummary(**prop)


def add_rms_information(
    field_summary: FieldSummary, aegean_outputs: AegeanOutputs
) -> FieldSummary:
    rms_image_path = aegean_outputs.rms

    rms_info = get_rms_image_info(rms_path=rms_image_path)
    centre = rms_info.centre.icrs
    telescope = field_summary.location
    ms_times = field_summary.ms_times
    centre_altaz = centre.transform_to(AltAz(obstime=ms_times, location=telescope))
    hour_angles = centre_altaz.az.to(u.hourangle)
    elevations = centre_altaz.alt.to(u.deg)

    no_components = len(Table.read(aegean_outputs.comp))

    field_summary = field_summary.with_options(
        centre=centre,
        hour_angles=hour_angles,
        elevations=elevations,
        no_components=no_components,
        median_rms=rms_info.median,
    )

    return field_summary


def create_field_summary(
    ms: Union[MS, Path],
    cal_sbid_path: Optional[Path] = None,
    holography_path: Optional[Path] = None,
    aegean_outputs: Optional[AegeanOutputs] = None,
) -> FieldSummary:
    logger.info("Creating field summary object")

    ms = MS.cast(ms=ms)

    sbid = str(get_sbid_from_path(path=ms.path))
    cal_sbid = str(get_sbid_from_path(path=cal_sbid_path)) if cal_sbid_path else None

    ms_times = get_times_from_ms(ms=ms)
    integration = ms_times.ptp().to(u.second).value
    location = get_telescope_location_from_ms(ms=ms)

    field_summary = FieldSummary(
        sbid=sbid,
        cal_sbid=cal_sbid,
        location=location,
        integration_time=integration,
        holography_path=holography_path,
        ms_times=Time([ms_times.min(), ms_times.max()]),
    )

    if aegean_outputs:
        field_summary = add_rms_information(
            field_summary=field_summary, aegean_outputs=aegean_outputs
        )

    return field_summary


def update_field_summary() -> FieldSummary:
    pass


# ms_times = get_times_from_ms(ms=ms)
#     telescope = get_telescope_location_from_ms(ms=ms)
#     centre_altaz = centre.transform_to(AltAz(obstime=ms_times, location=telescope))
#     hour_angles = centre_altaz.az.to(u.hourangle)
#     elevations = centre_altaz.alt.to(u.deg)
