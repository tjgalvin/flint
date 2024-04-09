from pathlib import Path

import pytest

from flint.configuration import (
    get_image_options_from_yaml,
    get_selfcal_options_from_yaml,
    create_default_yaml,
    load_yaml,
    verify_configuration,
    Strategy,
)


def test_create_yaml_file(tmpdir):
    output = create_default_yaml(
        output_yaml=Path(tmpdir) / "example.yaml", selfcal_rounds=3
    )

    assert output.exists()


def test_create_and_load(tmpdir):
    output = create_default_yaml(
        output_yaml=Path(tmpdir) / "example.yaml", selfcal_rounds=3
    )

    assert output.exists()

    strat = load_yaml(input_yaml=output)
    assert isinstance(strat, Strategy)

    strat = load_yaml(input_yaml=output, verify=False)
    assert isinstance(strat, Strategy)


def test_verify(tmpdir):
    output = create_default_yaml(
        output_yaml=Path(tmpdir) / "example.yaml", selfcal_rounds=3
    )

    assert output.exists()
    strat = load_yaml(input_yaml=output, verify=False)
    assert isinstance(strat, Strategy)

    _ = verify_configuration(input_config=strat)

    strat["ddd"] = 123
    with pytest.raises(ValueError):
        verify_configuration(input_config=strat)


def test_get_image_options():
    init = get_image_options_from_yaml(input_yaml=None, self_cal_rounds=False)

    assert isinstance(init, dict)

    rounds = get_image_options_from_yaml(input_yaml=None, self_cal_rounds=True)

    assert isinstance(init, dict)
    assert 1 in rounds.keys()


def test_raise_image_options_error():
    example = Path("example.yaml")

    with pytest.raises(AssertionError):
        get_image_options_from_yaml(input_yaml=example)


def test_self_cal_options():
    rounds = get_selfcal_options_from_yaml(input_yaml=None)

    assert isinstance(rounds, dict)
    assert 1 in rounds.keys()


def test_raise_error_options_error():
    example = Path("example.yaml")

    with pytest.raises(AssertionError):
        get_selfcal_options_from_yaml(input_yaml=example)
