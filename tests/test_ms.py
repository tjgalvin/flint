"""Some tests related to components around measurement sets. 
"""
from pathlib import Path
from flint.naming import get_sbid_from_path, raw_ms_format, RawNameComponents


def test_get_sbid_from_path():
    example_path = "/scratch3/gal16b/askap_sbids/39400/2022-04-14_100122_0.ms"

    sbid = get_sbid_from_path(path=Path(example_path))
    assert isinstance(sbid, int)
    assert sbid == 39400

    example_path_2 = "/scratch3/gal16b/askap_sbids/39400"
    sbid = get_sbid_from_path(path=Path(example_path_2))
    assert isinstance(sbid, int)
    assert sbid == 39400

    example_path_3 = (
        "/scratch3/gal16b/askap_sbids/39400/2022-04-14_100122_0.averaged.ms"
    )

    sbid = get_sbid_from_path(path=Path(example_path_3))
    assert isinstance(sbid, int)
    assert sbid == 39400


def test_raw_name_components():
    ms = "2022-04-14_100122_0.ms"

    components = raw_ms_format(ms)
    assert isinstance(components, RawNameComponents)
    assert components.beam == "0"
    assert components.date == "2022-04-14"
    assert components.time == "100122"
    assert components.spw == None

    ms = "2022-04-14_100122_0_3.ms"

    components = raw_ms_format(ms)
    assert isinstance(components, RawNameComponents)
    assert components.beam == "0"
    assert components.date == "2022-04-14"
    assert components.time == "100122"
    assert components.spw == "3"
