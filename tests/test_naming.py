"""Some tests related to components around measurement sets.
"""
from pathlib import Path

from flint.naming import (
    create_ms_name,
    get_sbid_from_path,
    raw_ms_format,
    processed_ms_format,
    extract_components_from_name,
    extract_beam_from_name,
    get_aocalibrate_output_path,
    RawNameComponents,
    ProcessedNameComponents,
)


def test_aocalibrate_into_process_ms_format():
    ms = Path("/some/made/up/path/SB123.Tim.beam33.round2.ms")

    outpath = get_aocalibrate_output_path(
        ms_path=ms, include_preflagger=True, include_smoother=True
    )
    named_components = processed_ms_format(in_name=outpath)
    assert isinstance(named_components, ProcessedNameComponents)
    assert named_components.sbid == "123"
    assert named_components.field == "Tim"
    assert named_components.beam == "33"


def test_aocalibrate_naming():
    ms = Path("/some/made/up/path/SB123.Tim.beam33.round2.ms")

    outpath = get_aocalibrate_output_path(
        ms_path=ms, include_preflagger=True, include_smoother=True
    )
    assert outpath == Path(
        "/some/made/up/path/SB123.Tim.beam33.aocalibrate.preflagged.smoothed.bin"
    )

    outpath = get_aocalibrate_output_path(
        ms_path=ms, include_preflagger=True, include_smoother=False
    )
    assert outpath == Path(
        "/some/made/up/path/SB123.Tim.beam33.aocalibrate.preflagged.bin"
    )

    outpath = get_aocalibrate_output_path(
        ms_path=ms, include_preflagger=False, include_smoother=False
    )
    assert outpath == Path("/some/made/up/path/SB123.Tim.beam33.aocalibrate.bin")

    processed_outputs = processed_ms_format(in_name=outpath)
    assert isinstance(processed_outputs, ProcessedNameComponents)

    ms = Path("/some/made/up/path/SB123.Tim.beam33.spw1.round2.ms")

    outpath = get_aocalibrate_output_path(
        ms_path=ms, include_preflagger=True, include_smoother=True
    )
    assert outpath == Path(
        "/some/made/up/path/SB123.Tim.beam33.spw1.aocalibrate.preflagged.smoothed.bin"
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


def test_components_all_beams_spws():
    for beam in range(36):
        for spw in range(10):
            ms = f"2022-04-14_100122_{beam}_{spw}.ms"

            components = raw_ms_format(ms)
            assert isinstance(components, RawNameComponents)
            assert components.beam == f"{beam}"
            assert components.date == "2022-04-14"
            assert components.time == "100122"
            assert components.spw == f"{spw}"

    for beam in range(36):
        ms = f"2022-04-14_100122_{beam}.ms"

        components = raw_ms_format(ms)
        assert isinstance(components, RawNameComponents)
        assert components.beam == f"{beam}"
        assert components.date == "2022-04-14"
        assert components.time == "100122"
        assert components.spw is None


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


def test_formatted_name_components():
    ex = "SB39400.RACS_0635-31.beam33-MFS-image.conv.fits"

    components = processed_ms_format(in_name=ex)
    assert isinstance(components, ProcessedNameComponents)
    assert components.sbid == "39400"
    assert components.field == "RACS_0635-31"
    assert components.beam == "33"
    assert components.spw is None
    assert components.round is None

    ex = "SB39400.RACS_0635-31.beam33.spw123-MFS-image.conv.fits"

    components = processed_ms_format(in_name=ex)
    assert isinstance(components, ProcessedNameComponents)
    assert components.sbid == "39400"
    assert components.field == "RACS_0635-31"
    assert components.beam == "33"
    assert components.spw == "123"
    assert components.round is None

    ex_path = Path(
        "/this/is/and/example/path/SB39400.RACS_0635-31.beam33.spw123-MFS-image.conv.fits"
    )

    components = processed_ms_format(in_name=ex_path)
    assert isinstance(components, ProcessedNameComponents)
    assert components.sbid == "39400"
    assert components.field == "RACS_0635-31"
    assert components.beam == "33"
    assert components.spw == "123"
    assert components.round is None


def test_formatted_name_components_wround():
    ex = "SB39400.RACS_0635-31.beam33.round1-MFS-image.conv.fits"

    components = processed_ms_format(in_name=ex)
    assert isinstance(components, ProcessedNameComponents)
    assert components.sbid == "39400"
    assert components.field == "RACS_0635-31"
    assert components.beam == "33"
    assert components.spw is None
    assert components.round == "1"

    ex = "SB39400.RACS_0635-31.beam33.spw123.round12-MFS-image.conv.fits"

    components = processed_ms_format(in_name=ex)
    assert isinstance(components, ProcessedNameComponents)
    assert components.sbid == "39400"
    assert components.field == "RACS_0635-31"
    assert components.beam == "33"
    assert components.spw == "123"
    assert components.round == "12"

    ex_path = Path(
        "/this/is/and/example/path/SB39400.RACS_0635-31.beam33.spw123.round123-MFS-image.conv.fits"
    )

    components = processed_ms_format(in_name=ex_path)
    assert isinstance(components, ProcessedNameComponents)
    assert components.sbid == "39400"
    assert components.field == "RACS_0635-31"
    assert components.beam == "33"
    assert components.spw == "123"
    assert components.round == "123"


def test_get_correct_name_format():
    examples = [
        "SB39400.RACS_0635-31.beam33-MFS-image.conv.fits",
        "SB39400.RACS_0635-31.beam33.ms",
        Path(
            "/this/is/and/example/path/SB39400.RACS_0635-31.beam33.spw123-MFS-image.conv.fits"
        ),
    ]

    for ex in examples:
        res = extract_components_from_name(name=ex)
        assert isinstance(res, ProcessedNameComponents)

    examples = ["2022-04-14_100122_0.averaged.ms"]

    for ex in examples:
        res = extract_components_from_name(name=ex)
        assert isinstance(res, RawNameComponents)


def test_get_beam_from_name():
    assert extract_beam_from_name(name="2022-04-14_100122_0.averaged.ms") == 0
    assert (
        extract_beam_from_name(name="SB39400.RACS_0635-31.beam33-MFS-image.conv.fits")
        == 33
    )
