"""Procedure to calibrate bandpass observation
"""
from argparse import ArgumentParser
from pathlib import Path

from flint.logging import logger
from flint.ms import MS
from flint.sky_model import KNOWN_1934_FILES, get_1934_model
from flint.calibrate.aocalibrate import calibrate_apply_ms


def calibrate_bandpass(
    ms_path: Path, data_column: str, mode: str, container: Path
) -> MS:
    logger.info(f"Will calibrate {str(ms_path)}, colum {data_column}")

    # TODO: Check to make sure only 1934-638
    model_path: Path = get_1934_model(mode=mode)

    ms = calibrate_apply_ms(
        ms_path=ms_path,
        model_path=model_path,
        container=container,
        data_column=data_column,
    )

    return ms


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Bandpass calibration procedure to calibrate a 1934-638 measurement set. "
    )

    supported_models = list(KNOWN_1934_FILES.keys())

    parser.add_argument("ms", type=Path, help="Path to the 1934-638 measurement set. ")
    parser.add_argument(
        "--data-column",
        str=str,
        default="DATA",
        help="Column containing the data to calibrate. ",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="calibrate",
        choices=supported_models,
        help=f"Support 1934-638 calibration models. available models: {supported_models}. ",
    )
    parser.add_argument(
        "--container",
        type=Path,
        default=Path("calibrate.sif"),
        help="Path to container that is capable or running and apply calibration solutions for desired mode. ",
    )

    return parser


def cli() -> None:
    import logging

    logger.setLevel(logging.INFO)

    parser = get_parser()

    args = parser.parse_args()


if __name__ == "__main__":
    cli()
