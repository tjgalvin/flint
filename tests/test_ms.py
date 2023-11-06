"""Some tests related to components around measurement sets. 
"""
from pathlib import Path
from flint.naming import (
    create_ms_name,
    get_sbid_from_path,
    raw_ms_format,
    RawNameComponents,
)


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
    assert components.spw is None

    ms = "2022-04-14_100122_0_3.ms"

    components = raw_ms_format(ms)
    assert isinstance(components, RawNameComponents)
    assert components.beam == "0"
    assert components.date == "2022-04-14"
    assert components.time == "100122"
    assert components.spw == "3"


def test_create_ms_name():
    example_path = "/scratch3/gal16b/askap_sbids/39400/2022-04-14_100122_0.ms"
    expected = "SB39400.beam0.ms"
    ms_path = create_ms_name(ms_path=example_path)
    assert isinstance(ms_path, str)
    assert ms_path == expected
    assert ms_path.endswith(".ms")

    example_path_2 = "/scratch3/gal16b/askap_sbids/39400/2022-04-14_100122_0.ms"
    expected_2 = "SB39400.RACS_0635-31.beam0.ms"
    ms_path_2 = create_ms_name(ms_path=example_path_2, field="RACS_0635-31")
    assert isinstance(ms_path_2, str)
    assert ms_path_2 == expected_2

    example_path_3 = "/scratch3/gal16b/askap_sbids/39400/2022-04-14_100122_0.ms"
    expected_3 = "SB1234.RACS_0635-31.beam0.ms"
    ms_path_3 = create_ms_name(ms_path=example_path_3, sbid=1234, field="RACS_0635-31")
    assert isinstance(ms_path_3, str)
    assert ms_path_3 == expected_3

    example_path_4 = "/scratch3/gal16b/askap_sbids/39400/2022-04-14_100122_0_12.ms"
    expected_4 = "SB1234.RACS_0635-31.beam0.spw12.ms"
    ms_path_4 = create_ms_name(ms_path=example_path_4, sbid=1234, field="RACS_0635-31")
    assert isinstance(ms_path_4, str)
    assert ms_path_4 == expected_4


def test_create_ms_name_no_sbid():
    example_path = "2022-04-14_100122_0.ms"
    expected = "SB0000.beam0.ms"
    ms_path = create_ms_name(ms_path=example_path)
    assert isinstance(ms_path, str)
    assert ms_path == expected

    expected_2 = "SB0000.RACS_0635-31.beam0.ms"
    ms_path_2 = create_ms_name(ms_path=example_path, field="RACS_0635-31")
    assert isinstance(ms_path_2, str)
    assert ms_path_2 == expected_2

    example_path_3 = "2022-04-14_100122_0_12.ms"
    expected_3 = "SB0000.RACS_0635-31.beam0.spw12.ms"
    ms_path_3 = create_ms_name(ms_path=example_path_3, field="RACS_0635-31")
    assert isinstance(ms_path_3, str)
    assert ms_path_3 == expected_3
