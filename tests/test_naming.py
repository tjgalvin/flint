"""Some tests related to components around measurement sets."""

from datetime import datetime
from pathlib import Path

import pytest

from flint.naming import (
    CASDANameComponents,
    FITSMaskNames,
    ProcessedNameComponents,
    RawNameComponents,
    add_timestamp_to_path,
    casda_ms_format,
    create_fits_mask_names,
    create_ms_name,
    extract_beam_from_name,
    extract_components_from_name,
    get_aocalibrate_output_path,
    get_beam_resolution_str,
    get_potato_output_base_path,
    get_sbid_from_path,
    get_selfcal_ms_name,
    processed_ms_format,
    raw_ms_format,
)


def test_get_beam_resolution_str():
    """Map the known / support modes of beam resolution in file names"""
    assert "raw" == get_beam_resolution_str(mode="raw")
    assert "optimal" == get_beam_resolution_str(mode="optimal")
    assert "fixed" == get_beam_resolution_str(mode="fixed")

    assert "raw!" == get_beam_resolution_str(mode="raw", marker="!")
    assert "optimal?" == get_beam_resolution_str(mode="optimal", marker="?")
    assert "fixed." == get_beam_resolution_str(mode="fixed", marker=".")

    with pytest.raises(ValueError):
        _ = get_beam_resolution_str("Jack")


def test_casda_ms_format_1934():
    """Checks around the name format form CASDA  considering 1934"""
    exs = [
        "1934.SB40470.beam35.ms",
        Path("1934.SB40470.beam35.ms"),
    ]
    for ex in exs:
        res = casda_ms_format(in_name=ex)
        assert res is not None
        assert res.format == "1934"
        assert res.sbid == 40470
        assert res.beam == "35"


def test_casda_ms_format():
    """Checks around the name format form CASDA"""
    exs = [
        "scienceData.RACS_1237+00.SB40470.RACS_1237+00.beam35_averaged_cal.leakage.ms",
        Path(
            "scienceData.RACS_1237+00.SB40470.RACS_1237+00.beam35_averaged_cal.leakage.ms"
        ),
    ]
    for ex in exs:
        res = casda_ms_format(in_name=ex)
        assert res is not None
        assert isinstance(res, CASDANameComponents)
        assert res.sbid == 40470
        assert res.beam == "35"
        assert res.field == "RACS_1237+00"
        assert res.alias == "RACS_1237+00"
        assert res.format == "scienceData"

    # Confirm None is returned in silly cases
    exs = [
        "scienceData.fdgdfdfg.RACS_1237+00.SBaveraged_cal.leakage.ms",
        "SB12349.RACS_1234+45.ms",
        "SB12349.RACS_1234+45.round2.ms",
    ]
    for ex in exs:
        res = casda_ms_format(in_name=ex)
        assert res is None


def test_self_cal_name_wbeams():
    """Checks around where the self-calibration naming is working, and the
    correct slicing. This test came after the one below and includes the beam
    number as a component to consider"""

    for beam in range(45):
        ms = Path(f"SB12349.RACS_1234+45.beam{beam:02d}.ms")
        e_ms = Path(f"SB12349.RACS_1234+45.beam{beam:02d}.round1.ms")
        out_ms = get_selfcal_ms_name(in_ms_path=ms, round=1)
        assert out_ms == e_ms

    for round in range(1, 5):
        for beam in range(45):
            ms = Path(f"SB12349.RACS_1234+45.beam{beam:02d}.round{round}.ms")
            e_ms = Path(f"SB12349.RACS_1234+45.beam{beam:02d}.round{round+1}.ms")
            out_ms = get_selfcal_ms_name(in_ms_path=ms, round=round + 1)
            assert out_ms == e_ms


def test_self_cal_name():
    """Checks around where the self-calibration naming is working, and the
    correct slicing"""

    ms = Path("SB12349.RACS_1234+45.ms")
    e_ms = Path("SB12349.RACS_1234+45.round1.ms")
    out_ms = get_selfcal_ms_name(in_ms_path=ms, round=1)
    assert out_ms == e_ms

    ms = Path("SB12349.RACS_1234+45.round1.ms")
    e_ms = Path("SB12349.RACS_1234+45.round2.ms")
    out_ms = get_selfcal_ms_name(in_ms_path=ms, round=2)
    assert out_ms == e_ms

    ms = Path(
        "/some/other/directory/SB12349.RACS_1234+45.verrrrryyylonnnnnnnhhhhh.round1.ms"
    )
    e_ms = Path(
        "/some/other/directory/SB12349.RACS_1234+45.verrrrryyylonnnnnnnhhhhh.round2.ms"
    )
    out_ms = get_selfcal_ms_name(in_ms_path=ms, round=2)
    assert out_ms == e_ms


def test_potato_output_name():
    ms = Path("/some/made/up/path/SB123.Tim.beam33.round2.ms")
    potato = get_potato_output_base_path(ms_path=ms)

    assert isinstance(potato, Path)
    assert potato == Path("/some/made/up/path/SB123.Tim.beam33.potato")

    config = potato.parent / (potato.name + ".config")
    assert config == Path("/some/made/up/path/SB123.Tim.beam33.potato.config")


def test_add_timestamp_to_path():
    # make sure adding a timestamp to a file name works
    dd = datetime(2024, 4, 12, 10, 30, 50, 243910)

    example_str = "/test/this/is/filename.txt"
    stamped_path = add_timestamp_to_path(input_path=example_str, timestamp=dd)
    expected = Path("/test/this/is/filename-20240412-103050.txt")

    assert stamped_path == expected

    example_path = Path("/test/this/is/filename.txt")
    stamped_path = add_timestamp_to_path(input_path=example_path, timestamp=dd)

    assert stamped_path == expected

    now_path = add_timestamp_to_path(input_path=example_path)
    assert now_path != expected


def test_create_fits_mask_names_no_signal():
    fits_image = Path("38960/SB38960.RACS_1418-12.noselfcal_linmos.fits")

    result = create_fits_mask_names(fits_image=fits_image, include_signal_path=False)
    assert isinstance(result, FITSMaskNames)
    assert isinstance(result.mask_fits, Path)
    assert result.signal_fits is None
    assert result.mask_fits == Path(
        "38960/SB38960.RACS_1418-12.noselfcal_linmos.mask.fits"
    )


def test_create_fits_mask_names():
    fits_image = Path("38960/SB38960.RACS_1418-12.noselfcal_linmos.fits")

    result = create_fits_mask_names(fits_image=fits_image, include_signal_path=True)
    assert isinstance(result, FITSMaskNames)
    assert isinstance(result.signal_fits, Path)
    assert isinstance(result.mask_fits, Path)
    assert result.signal_fits == Path(
        "38960/SB38960.RACS_1418-12.noselfcal_linmos.signal.fits"
    )
    assert result.mask_fits == Path(
        "38960/SB38960.RACS_1418-12.noselfcal_linmos.mask.fits"
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
    expected = "SB39400.beam00.ms"
    ms_path = create_ms_name(ms_path=example_path)
    assert isinstance(ms_path, str)
    assert ms_path == expected
    assert ms_path.endswith(".ms")

    example_path_2 = "/scratch3/gal16b/askap_sbids/39400/2022-04-14_100122_0.ms"
    expected_2 = "SB39400.RACS_0635-31.beam00.ms"
    ms_path_2 = create_ms_name(ms_path=example_path_2, field="RACS_0635-31")
    assert isinstance(ms_path_2, str)
    assert ms_path_2 == expected_2

    example_path_3 = "/scratch3/gal16b/askap_sbids/39400/2022-04-14_100122_0.ms"
    expected_3 = "SB1234.RACS_0635-31.beam00.ms"
    ms_path_3 = create_ms_name(ms_path=example_path_3, sbid=1234, field="RACS_0635-31")
    assert isinstance(ms_path_3, str)
    assert ms_path_3 == expected_3

    example_path_4 = "/scratch3/gal16b/askap_sbids/39400/2022-04-14_100122_0_12.ms"
    expected_4 = "SB1234.RACS_0635-31.beam00.spw12.ms"
    ms_path_4 = create_ms_name(ms_path=example_path_4, sbid=1234, field="RACS_0635-31")
    assert isinstance(ms_path_4, str)
    assert ms_path_4 == expected_4

    examples = [
        "scienceData.RACS_1237+00.SB40470.RACS_1237+00.beam35_averaged_cal.leakage.ms",
    ]
    for ex in examples:
        name = create_ms_name(ms_path=ex)
        assert name == "SB40470.RACS_1237+00.beam35.ms"


def test_create_ms_name_no_sbid():
    example_path = "2022-04-14_100122_0.ms"
    expected = "SB0000.beam00.ms"
    ms_path = create_ms_name(ms_path=example_path)
    assert isinstance(ms_path, str)
    assert ms_path == expected

    expected_2 = "SB0000.RACS_0635-31.beam00.ms"
    ms_path_2 = create_ms_name(ms_path=example_path, field="RACS_0635-31")
    assert isinstance(ms_path_2, str)
    assert ms_path_2 == expected_2

    example_path_3 = "2022-04-14_100122_0_12.ms"
    expected_3 = "SB0000.RACS_0635-31.beam00.spw12.ms"
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

    examples = [
        "scienceData.RACS_1237+00.SB40470.RACS_1237+00.beam35_averaged_cal.leakage.ms",
        Path(
            "scienceData.RACS_1237+00.SB40470.RACS_1237+00.beam35_averaged_cal.leakage.ms"
        ),
    ]
    for ex in examples:
        res = extract_components_from_name(name=ex)
        assert isinstance(res, CASDANameComponents)


def test_get_beam_from_name():
    assert extract_beam_from_name(name="2022-04-14_100122_0.averaged.ms") == 0
    assert (
        extract_beam_from_name(name="SB39400.RACS_0635-31.beam33-MFS-image.conv.fits")
        == 33
    )
