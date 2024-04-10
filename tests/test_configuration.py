from pathlib import Path

import pytest

from flint.configuration import (
    get_image_options_from_yaml,
    get_selfcal_options_from_yaml,
    create_default_yaml,
    load_yaml,
    verify_configuration,
    get_options_from_strategy,
    Strategy,
)
from flint.utils import get_packaged_resource_path


@pytest.fixture
def package_strategy():
    example = get_packaged_resource_path(
        package="flint", filename="data/tests/test_config.yaml"
    )

    strategy = load_yaml(input_yaml=example, verify=False)

    return strategy


@pytest.fixture
def strategy(tmpdir):
    output = create_default_yaml(
        output_yaml=Path(tmpdir) / "example.yaml", selfcal_rounds=3
    )

    strat = load_yaml(input_yaml=output, verify=False)

    return strat


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
    # test to make sure we can generate a default strategy (see pytest fixture)
    # read it backinto a dict and verify it is valid
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


def test_get_options(strategy):
    # test to make sure we can generate a default strategy (see pytest fixture)
    # read it backinto a dict and then access some attributes
    wsclean = get_options_from_strategy(
        strategy=strategy, mode="wsclean", round="initial"
    )
    assert isinstance(wsclean, dict)
    # example options
    assert wsclean["data_column"] == "CORRECTED_DATA"

    wsclean = get_options_from_strategy(strategy=strategy, mode="wsclean", round=1)
    assert isinstance(wsclean, dict)
    # example options
    assert wsclean["data_column"] == "CORRECTED_DATA"


def test_get_modes(package_strategy):
    # makes sure defaults for these modes are return when reuestion options
    # on a self-cal round without them set
    strategy = package_strategy

    for mode in ("wsclean", "gaincal", "masking"):
        test = get_options_from_strategy(strategy=strategy, mode=mode, round=1)
        assert isinstance(test, dict)
        assert len(test.keys()) > 0


def test_bad_round(package_strategy):
    # make sure incorrect round is handled properly
    with pytest.raises(AssertionError):
        _ = get_options_from_strategy(strategy=package_strategy, round="doesnotexists")

    with pytest.raises(AssertionError):
        _ = get_options_from_strategy(strategy=package_strategy, round=1.23456)


def test_empty_strategy_empty_options():
    # if None is given as a strategy state then empty set of options is return
    res = get_options_from_strategy(strategy=None)

    assert isinstance(res, dict)
    assert not res == 0


def test_max_round_override(package_strategy):
    # ebsyre that the logic to switch to the highest available slefcal
    # round is sound
    strategy = package_strategy

    opts = get_options_from_strategy(strategy=strategy, round=9999)
    assert isinstance(opts, dict)
    assert opts["data_column"] == "TheLastRoundIs3"


def test_empty_options(package_strategy):
    # ensures an empty options dict is return should something not set is requested
    items = get_options_from_strategy(
        strategy=package_strategy, mode="thisdoesnotexist"
    )

    assert isinstance(items, dict)
    assert len(items.keys()) == 0


def test_assert_strategy_bad():
    # tests to see if a error is raised when a bad strategy instance is passed in
    with pytest.raises(AssertionError):
        _ = get_options_from_strategy(strategy="ThisIsBad")


def test_updated_get_options(package_strategy):

    strategy = package_strategy
    assert isinstance(strategy, Strategy)

    wsclean_init = get_options_from_strategy(
        strategy=strategy, mode="wsclean", round="initial"
    )
    assert wsclean_init["data_column"] == "CORRECTED_DATA"

    wsclean_1 = get_options_from_strategy(strategy=strategy, mode="wsclean", round=1)
    assert wsclean_init["data_column"] == "CORRECTED_DATA"
    assert wsclean_1["data_column"] == "EXAMPLE"

    wsclean_2 = get_options_from_strategy(strategy=strategy, mode="wsclean", round=2)
    assert wsclean_2["multiscale"] is False
    assert wsclean_2["data_column"] == "CORRECTED_DATA"

    assert all(
        [
            wsclean_init[key] == wsclean_1[key]
            for key in wsclean_init.keys()
            if key != "data_column"
        ]
    )
    assert wsclean_init["data_column"] != wsclean_1["data_column"]

    assert all(
        [
            wsclean_init[key] == wsclean_2[key]
            for key in wsclean_init.keys()
            if key != "multiscale"
        ]
    )


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
