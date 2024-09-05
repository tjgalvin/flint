"""Tests around BANE and aegean, which are one of the source finding
tools used in flint. BANE is also used to derive signal maps that ultimately
feed the clean mask creation.
"""

from pathlib import Path

from flint.source_finding.aegean import (
    AegeanOptions,
    BANEOptions,
    _get_aegean_command,
    _get_bane_command,
)


def test_bane_options():
    bane_opts = BANEOptions(box_size=(2, 1), grid_size=(20, 10))

    assert isinstance(bane_opts, BANEOptions)

    bane_str = _get_bane_command(
        image=Path("this/is/a/test.fits"), cores=8, bane_options=bane_opts
    )

    assert isinstance(bane_str, str)

    expected = "BANE this/is/a/test.fits --cores 8 --stripes 7 --grid 20 10 --box 2 1"
    assert expected == bane_str


def test_bane_options_with_defaults():
    bane_opts = BANEOptions(box_size=None, grid_size=None)

    assert isinstance(bane_opts, BANEOptions)

    bane_str = _get_bane_command(
        image=Path("this/is/a/test.fits"), cores=8, bane_options=bane_opts
    )

    assert isinstance(bane_str, str)

    expected = "BANE this/is/a/test.fits --cores 8 --stripes 7"
    assert expected == bane_str

    bane_opts = BANEOptions(box_size=(3, 5), grid_size=None)
    bane_str = _get_bane_command(
        image=Path("this/is/a/test.fits"), cores=8, bane_options=bane_opts
    )
    expected = "BANE this/is/a/test.fits --cores 8 --stripes 7 --box 3 5"
    assert expected == bane_str


def test_aegean_options():
    """Testing basic aegean options creation and expected command"""
    aegean_options = AegeanOptions(nocov=True, maxsummits=4, autoload=True)

    # This are silly tests
    assert isinstance(aegean_options, AegeanOptions)
    assert aegean_options.maxsummits == 4

    ex_path = Path("this/is/a/test.fits")
    aegean_command = _get_aegean_command(
        image=ex_path, base_output="example", aegean_options=aegean_options
    )

    expected_cmd = "aegean this/is/a/test.fits --autoload --nocov --maxsummits 4 --table example.fits"
    assert aegean_command == expected_cmd

    aegean_options2 = AegeanOptions(nocov=False, maxsummits=40, autoload=False)
    aegean_command2 = _get_aegean_command(
        image=ex_path, base_output="example", aegean_options=aegean_options2
    )

    expected_cmd2 = "aegean this/is/a/test.fits --maxsummits 40 --table example.fits"
    assert aegean_command2 == expected_cmd2
