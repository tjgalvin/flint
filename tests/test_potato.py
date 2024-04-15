"""Some basic checks around the potato peel functionality"""

from astropy.table import Table

from flint.peel.potato import load_known_peel_sources


def test_load_peel_sources():
    """Ensure we can load in the reference set of sources for peeling"""

    tab = load_known_peel_sources()
    assert isinstance(tab, Table)
    assert len(tab) > 4
