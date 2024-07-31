"""Tests that work around the catalogue functionality"""

import pytest
from pathlib import Path

from astropy.table import Table

from flint.catalogue import (
    download_vizier_catalogue,
    download_referencce_catalogues,
    get_reference_catalogue,
    KNOWN_REFERENCE_CATALOGUES,
)


def test_known_reference_catalogues():
    """Make sure all of the known reference catalogues have a vizier id attached"""
    assert all([cata.vizier_id for cata in KNOWN_REFERENCE_CATALOGUES.values()])


def test_download_vizier_catalogue(tmpdir):
    """Download a example vizier table"""

    output_path = Path(tmpdir) / "catalogue1/ICRF.fits"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    icrf_id = KNOWN_REFERENCE_CATALOGUES["ICRF"]
    assert icrf_id.vizier_id
    cata_path = download_vizier_catalogue(
        output_path=output_path, vizier_id=icrf_id.vizier_id
    )

    assert cata_path == output_path
    assert cata_path.exists()

    table = Table.read(cata_path)
    assert len(table) == 3414


def test_get_vizier_catalogue(tmpdir):
    """Download a example vizier table"""
    output_path = Path(tmpdir) / "catalogue1/ICRF.fits"
    assert not output_path.exists()

    output_path = Path(tmpdir) / "catalogue1/ICRF.fits"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    icrf = KNOWN_REFERENCE_CATALOGUES["ICRF"]
    assert icrf.vizier_id
    _ = download_vizier_catalogue(output_path=output_path, vizier_id=icrf.vizier_id)
    assert output_path.exists()

    table, catalogue = get_reference_catalogue(
        reference_directory=output_path.parent, survey="ICRF"
    )
    assert catalogue.file_name == "ICRF.fits"
    assert catalogue.survey == "ICRF"
    assert len(table) == 3414

    with pytest.raises(ValueError):
        _, _ = get_reference_catalogue(
            reference_directory=output_path.parent, survey="Jack"
        )


def test_download_vizier_catalogue_dryrun(tmpdir):
    """See if the dry run option in download a example vizier table"""

    output_path = Path(tmpdir) / "cataloguedry/ICRF.fits"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    icrf_id = KNOWN_REFERENCE_CATALOGUES["ICRF"]

    cata_path = download_vizier_catalogue(
        output_path=output_path, vizier_id=icrf_id, dry_run=True
    )

    assert cata_path == output_path
    assert not cata_path.exists()


def test_download_reference_catalogues(tmpdir):
    """Ensure all catalogues can be downloaded. Not the dry_run=True,
    meaning the catalogues are not all actually dowenloaded
    """
    output_dir = Path(tmpdir) / "catalogue2"
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = download_referencce_catalogues(
        reference_directory=output_dir, dry_run=True
    )

    assert len(outputs) == len(KNOWN_REFERENCE_CATALOGUES)