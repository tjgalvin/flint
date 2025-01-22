"""Testing components in the leakage creation steps"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table
from astropy.wcs import WCS

from flint.leakage import (
    FITSImage,
    LeakageFilters,
    PixelCoords,
    _get_output_catalogue_path,
    _load_component_table,
    _load_fits_image,
    filter_components,
    get_xy_pixel_coords,
)
from flint.utils import get_packaged_resource_path


def test_get_catalogue_output_path():
    """Make sure the name of the catalogue comes out as expected"""
    catalogue = Path("SB1234.JACK_1234+56.i.round2_comp.fits")
    pol = "v"

    output_path = _get_output_catalogue_path(input_path=catalogue, pol=pol)
    assert output_path == Path("SB1234.JACK_1234+56.i.round2_comp.v_leakage.fits")

    spec_path = Path("JackSparrow_leakage.fits")
    output_path = _get_output_catalogue_path(
        input_path=catalogue, pol=pol, output_path=spec_path
    )
    assert output_path == spec_path

    with pytest.raises(AssertionError):
        _get_output_catalogue_path(input_path="notapathjack", pol=pol)


def _get_aegean_catalogue():
    table_path = get_packaged_resource_path(
        package="flint.data.tests",
        filename="SB38959.RACS_1357-18.noselfcal.linmos_comp.fits",
    )
    table = Table.read(table_path)
    return table, table_path


def _get_fits_image():
    fits_path = get_packaged_resource_path(
        package="flint.data.tests",
        filename="SB56659.RACS_0940-04.beam17.round3-0000-image.sub.fits",
    )

    return fits_path


def test_get_xy_coords():
    """Test on the conversion of component positions to pixel (x,y)"""
    fits_path = _get_fits_image()
    table, _ = _get_aegean_catalogue()

    fits_image = _load_fits_image(fits_path=fits_path)
    pixel_coords = get_xy_pixel_coords(table=table, wcs=fits_image.wcs)

    assert isinstance(pixel_coords, PixelCoords)
    assert pixel_coords.x.dtype == int
    assert pixel_coords.y.dtype == int
    assert len(pixel_coords.x) == len(pixel_coords.y)
    assert len(pixel_coords.x) == len(table)


def test_filter_component_catalogue():
    """Attempt to filter components from a catalogue"""
    table, _ = _get_aegean_catalogue()

    leakage_filters = LeakageFilters(
        isolation_radius_deg=0.0155,
        upper_int_peak_ratio=1.2,
        lower_int_peak_ratio=0.8,
        source_snr=40,
    )

    subset_table = filter_components(
        table=table,
        peak_col="peak_flux",
        int_col="int_flux",
        int_err_col="local_rms",
        leakage_filters=leakage_filters,
    )
    assert (
        len(subset_table) == 523
    )  # Did this in an interactive shell to get the expected.

    with pytest.raises(AssertionError):
        filter_components(
            table=table,
            peak_col="JackSparrow",
            int_col="int_flux",
            int_err_col="local_rms",
            leakage_filters=leakage_filters,
        )


def test_load_component_catalogue():
    """Test loading a component catalogue"""
    table, table_path = _get_aegean_catalogue()
    assert isinstance(table_path, Path)

    return_table = _load_component_table(catalogue=table)
    assert return_table is table

    loaded_table = _load_component_table(catalogue=table_path)
    assert loaded_table is not Table
    assert loaded_table.colnames == return_table.colnames


def test_load_fits_image():
    """Testinging the load function that attempts to read a FITS image and extract its bits"""
    fits_path = _get_fits_image()

    assert isinstance(fits_path, Path)
    assert fits_path.exists()

    fits_image = _load_fits_image(fits_path=fits_path)

    assert isinstance(fits_image, FITSImage)
    assert fits_image.path == fits_path
    assert isinstance(fits_image.data, np.ndarray)
    assert isinstance(fits_image.wcs, WCS)
    assert isinstance(fits_image.header, dict)

    with pytest.raises(AssertionError):
        _load_fits_image(fits_path=fits_path.with_suffix(".jack.sparrow"))
