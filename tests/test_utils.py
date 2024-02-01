"""Basic tests for utility functions
"""

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

from flint.utils import estimate_skycoord_centre


def test_estimate_skycoord_centre():
    ras = np.arange(-3, 3, 1) + 180.0
    decs = np.arange(-3, 3, 1) - 40.0

    sky_pos = SkyCoord(ras, decs, unit=(u.deg, u.deg))

    mean_pos = estimate_skycoord_centre(sky_positions=sky_pos)

    print(mean_pos)

    assert np.isclose(mean_pos.ra.deg, 179.54350474)
    assert np.isclose(mean_pos.dec.deg, -40.51256163)


def test_estimate_skycoord_centre_wrap():
    ras = np.arange(-3, 3, 1) + 360.0 % 360
    decs = np.arange(-3, 3, 1) - 40.0

    sky_pos = SkyCoord(ras, decs, unit=(u.deg, u.deg))

    mean_pos = estimate_skycoord_centre(sky_positions=sky_pos)

    print(mean_pos)

    assert np.isclose(mean_pos.ra.deg, 359.54349533)
    assert np.isclose(mean_pos.dec.deg, -40.51255648)
