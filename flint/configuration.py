"""Basic utilities to load operational parameters from a yaml-based
configuration file. The idea being that a configuration file would
be used to specify the options for imaging and self-calibration
throughout the pipeline.
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from flint.imager.wsclean import WSCleanOptions
from flint.logging import logger
from flint.masking import MaskingOptions
from flint.selfcal.casa import GainCalOptions

KNOWN_HEADERS = ("defaults", "initial", "selfcal", "version")
FORMAT_VERSION = 0.1


# A simple representation to pass around. Will help the type
# analysis and future pirates be clear with their mutinous
# intentions
class Strategy(dict):
    """Base representation for handling a loaded flint
    strategy"""

    pass


def get_selfcal_options_from_yaml(input_yaml: Optional[Path] = None) -> Dict:
    """Stub to represent interaction with a configurationf ile

    If a path is supplied, an error is raised.

    Args:
        input_yaml (Optional[Path], optional): Path to the configuration file. . Defaults to Optional[Path]=None.

    Returns:
        Dict: Mapping where the key is the self-calibration round, and values are key-value of updated gaincal options
    """

    assert (
        input_yaml is None
    ), "Configuring via a yaml configuration file is not yet support. "

    return {
        1: {"solint": "60s", "uvrange": ">235m", "nspw": 1},
        2: {"solint": "30s", "calmode": "p", "uvrange": ">235m", "nspw": 4},
        3: {"solint": "60s", "calmode": "ap", "uvrange": ">235m", "nspw": 4},
        4: {"solint": "30s", "calmode": "ap", "uvrange": ">235m", "nspw": 4},
    }


def get_image_options_from_yaml(
    input_yaml: Optional[Path] = None, self_cal_rounds: bool = False
) -> Dict:
    """Stub to interact with configuration file.

    If a `input_yaml` file is provided an error is raised

    Args:
        input_yaml (Optional[Path], optional): Should be None. Defaults to None.
        self_cal_rounds (bool, optional): Whether options for first imaging is being provided, or options to supply for each self-cal round. Defaults to False.

    Returns:
        Dict: _description_
    """

    assert (
        input_yaml is None
    ), "Configuring via a yaml configuration file is not yet support. "

    MULTISCALE_SCALES = (0, 15, 30, 40, 50, 60, 70, 120)
    IMAGE_SIZE = 7144

    if not self_cal_rounds:
        return {
            "size": IMAGE_SIZE,
            "minuvw_m": 235,
            "weight": "briggs -1.5",
            "scale": "2.5arcsec",
            "nmiter": 10,
            "force_mask_rounds": 10,
            "deconvolution_channels": 4,
            "fit_spectral_pol": 3,
            "auto_mask": 10,
            "multiscale": True,
            "local_rms_window": 55,
            "multiscale_scales": MULTISCALE_SCALES,
        }
    else:
        return {
            1: {
                "size": IMAGE_SIZE,
                "weight": "briggs -1.5",
                "scale": "2.5arcsec",
                "nmiter": 20,
                "force_mask_rounds": 8,
                "minuvw_m": 235,
                "channels_out": 18,
                "deconvolution_channels": 6,
                "fit_spectral_pol": 3,
                "auto_mask": 8.0,
                "local_rms_window": 55,
                "multiscale_scales": MULTISCALE_SCALES,
            },
            2: {
                "size": IMAGE_SIZE,
                "weight": "briggs -1.5",
                "scale": "2.5arcsec",
                "multiscale": True,
                "minuvw_m": 235,
                "nmiter": 20,
                "force_mask_rounds": 8,
                "channels_out": 18,
                "deconvolution_channels": 6,
                "fit_spectral_pol": 3,
                "auto_mask": 7.0,
                "local_rms_window": 55,
                "multiscale_scales": MULTISCALE_SCALES,
            },
            3: {
                "size": IMAGE_SIZE,
                "weight": "briggs -1.5",
                "scale": "2.5arcsec",
                "multiscale": True,
                "minuvw_m": 235,
                "nmiter": 20,
                "force_mask_rounds": 8,
                "channels_out": 18,
                "deconvolution_channels": 6,
                "fit_spectral_pol": 3,
                "auto_mask": 6.0,
                "local_rms_window": 55,
                "multiscale_scales": MULTISCALE_SCALES,
            },
            4: {
                "size": IMAGE_SIZE,
                "weight": "briggs -1.5",
                "scale": "2.5arcsec",
                "multiscale": True,
                "minuvw_m": 235,
                "nmiter": 20,
                "force_mask_rounds": 10,
                "channels_out": 16,
                "deconvolution_channels": 4,
                "fit_spectral_pol": 3,
                "auto_mask": 8,
                "local_rms_window": 55,
                "multiscale_scales": MULTISCALE_SCALES,
            },
            5: {
                "size": IMAGE_SIZE,
                "weight": "briggs -1.5",
                "scale": "2.5arcsec",
                "multiscale": True,
                "minuvw_m": 235,
                "nmiter": 20,
                "force_mask_rounds": 10,
                "channels_out": 4,
                "fit_spectral_pol": 3,
                "auto_mask": 7.0,
                "local_rms_window": 55,
                "multiscale_scales": MULTISCALE_SCALES,
            },
        }


def get_options_from_strategy(
    strategy: Strategy, mode: str = "wsclean", round: Union[str, int] = "initial"
) -> Dict[Any, Any]:
    """Extract a set of options from a strategy file to use in a pipeline
    run. If the mode exists in the default section, these are used as a base.

    If the mode exists and a round is specified, the options listed in the
    round are used to update the defaults.

    Args:
        strategy (Strategy): A loaded instance of a strategy file
        mode (str, optional): Which set of options to load. Typical values are `wsclean`, `gaincal` and `masking`. Defaults to "wsclean".
        round (Union[str, int], optional): Which round to load options for. May be `initial` or an `int` (which indicated a self-calibration round). Defaults to "initial".

    Raises:
        ValueError: An unrecongised value for `round`.
        AssertError: An unrecongised value for `round`.

    Returns:
        Dict[Any, Any]: Options specific to the requested set
    """

    # Some sanity checks
    assert isinstance(
        strategy, (Strategy, dict)
    ), f"Unknown input strategy type {type(strategy)}"
    assert round == "initial" or isinstance(
        round, int
    ), f"{round=} not a known value or type. "

    # step one, get the defaults
    options = (
        dict(**strategy["defaults"][mode])
        if mode in strategy["defaults"].keys()
        else {}
    )
    logger.debug(f"Defaults for {mode=}, {options=}")

    # Now get the updates
    if round == "initial":
        # separate function to avoid a missing mode from raising valu error
        if mode in strategy["initial"].keys():
            update_options = dict(**strategy["initial"][mode])
            logger.debug(f"Updating options with {update_options=}")
            options.update(update_options)
    elif isinstance(round, int):
        # separate function to avoid a missing mode from raising valu error
        if (
            round in strategy["selfcal"].keys()
            and mode in strategy["selfcal"][round].keys()
        ):
            update_options = dict(**strategy["selfcal"][round][mode])
            logger.debug(f"Updating options with {update_options=}")
            options.update(update_options)
    else:
        raise ValueError(f"{round=} not recognised.")

    return options


def verify_configuration(input_config: Strategy, raise_on_error: bool = True) -> bool:
    """Perform basic checks on the configuration file

    Args:
        input_config (Strategy): The loaded configuraiton file structure
        raise_on_error (bool, optional): Whether to raise an error should an issue in thew config file be found. Defaults to True.

    Raises:
        ValueError: Whether structure is valid

    Returns:
        bool: Config file is not valid. Raised only if `raise_on_error` is `True`
    """

    errors = []

    unknown_headers = [
        header for header in input_config.keys() if header not in KNOWN_HEADERS
    ]
    if unknown_headers:
        errors.append(f"{unknown_headers=} found. Supported headers: {KNOWN_HEADERS}")

    if "initial" not in input_config.keys():
        errors.append("No initial imaging round parameters")

    if "selfcal" in input_config.keys():
        round_keys = input_config["selfcal"].keys()

        if not all([isinstance(i, int) for i in round_keys]):
            errors.append("The keys into the self-calibration should be ints. ")

    valid_config = len(errors) == 0
    if not valid_config:
        for error in errors:
            logger.warning(error)

        if raise_on_error:
            raise ValueError("Configuration file not valid. ")

    return valid_config


def load_yaml(input_yaml: Path, verify: bool = True) -> Strategy:
    """Load in a flint based configuration file, which
    will be used to form the strategy for imaging of
    a field.

    The format of the return is likely to change. This
    is not to be relied on for the moment, and should
    be considered a toy. There will be a mutiny.

    Args:
        input_yaml (Path): The imaging strategy to use
        verify (bool, optional): Apply some basic checks to ensure a correctly formed strategy. Defaults to True.

    Returns:
        Strategy: The parameters of the imaging and self-calibration to use.
    """

    logger.info(f"Loading {input_yaml} file. ")

    with open(input_yaml, "r") as in_file:
        input_strategy = Strategy(yaml.load(in_file, Loader=yaml.Loader))

    if verify:
        verify_configuration(input_config=input_strategy)

    return input_strategy


def create_default_yaml(
    output_yaml: Path, selfcal_rounds: Optional[int] = None
) -> Path:
    """Create an example stategy yaml file that outlines the options to use at varies stages
    of some assumed processing pipeline.

    This is is completely experimental, and expected fields might change.

    Args:
        output_yaml (Path): Location to write the yaml file to.
        selfcal_rounds (Optional[int], optional): Will specify the number of self-calibration loops to include the file. If None, there will be none written. Defaults to None.

    Returns:
        Path: Path to the written yaml output file.
    """
    logger.info("Generating a default stategy. ")
    strategy: Dict[Any, Any] = {}

    strategy["version"] = FORMAT_VERSION

    strategy["defaults"] = {
        "wsclean": WSCleanOptions()._asdict(),
        "gaincal": GainCalOptions()._asdict(),
        "masking": MaskingOptions()._asdict(),
    }

    strategy["initial"] = {"wsclean": {}}

    if selfcal_rounds:
        logger.info(f"Creating {selfcal_rounds} self-calibration rounds. ")
        selfcal = {}
        for round in range(1, selfcal_rounds + 1):
            selfcal[round] = {
                "wsclean": {},
                "gaincal": {},
                "masking": {},
            }

        strategy["selfcal"] = selfcal

    with open(output_yaml, "w") as out_file:
        logger.info(f"Writing {output_yaml}.")
        yaml.dump(data=strategy, stream=out_file, sort_keys=False)

    return output_yaml


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Tooling to interact with flint yaml configuration files. "
    )

    subparser = parser.add_subparsers(dest="mode")

    create_parser = subparser.add_parser(
        "create", help="Create an initail yaml file for editing. "
    )
    create_parser.add_argument(
        "output_yaml",
        type=Path,
        default="flint_strategy.yaml",
        help="The output YAML file to write with default options for various stages. ",
    )
    create_parser.add_argument(
        "--selfcal-rounds",
        type=int,
        default=None,
        help="Number of self-calibration rounds to use. ",
    )

    load_parser = subparser.add_parser("load")
    load_parser.add_argument(
        "input_yaml",
        type=Path,
        help="Path to a strategy yaml file to load and inspect. ",
    )
    verify_parser = subparser.add_parser(
        "verify", help="Verify a yaml file is correct, as far as we can tell.  "
    )
    verify_parser.add_argument("input_yaml", type=Path, help="Path to a strategy file")

    return parser


def cli() -> None:
    parser = get_parser()

    args = parser.parse_args()

    if args.mode == "create":
        create_default_yaml(
            output_yaml=args.output_yaml, selfcal_rounds=args.selfcal_rounds
        )
    elif args.mode == "load":
        load_yaml(input_yaml=args.input_yaml)
    elif args.mode == "verify":
        input_config = load_yaml(input_yaml=args.input_yaml)
        if verify_configuration(input_config=input_config):
            logger.info(f"{args.input_yaml} appears valid")
    else:
        logger.error(f"{args.mode=} is not set or not known. Check --help. ")


if __name__ == "__main__":
    cli()
