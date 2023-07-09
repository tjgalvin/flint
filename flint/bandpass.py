"""Procedure to calibrate bandpass observation
"""
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from flint.logging import logger
from flint.ms import MS
from flint.sky_model import KNOWN_1934_FILES, get_1934_model
from flint.calibrate.aocalibrate import calibrate_apply_ms, AOSolutions


def plot_solutions(solutions_path: Path, ref_ant: int = 0) -> None:
    """Plot solutions for AO-style solutions

    Args:
        solutions_path (Path): Path to the solutions file
        ref_ant (int, optional): Reference antenna to use. Defaults to 0.
    """
    logger.info(f"Plotting {solutions_path}")

    ao_sols = AOSolutions.load(path=solutions_path)
    plot_paths = ao_sols.plot_solutions(ref_ant=ref_ant)


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

    subparser = parser.add_subparsers(
        dest="mode", help="Operation mode of flint_bandpass"
    )
    band_parser = subparser.add_parser("calibrate", help="Perform bandpass calibration")

    supported_models = list(KNOWN_1934_FILES.keys())

    band_parser.add_argument(
        "ms", type=Path, help="Path to the 1934-638 measurement set. "
    )
    band_parser.add_argument(
        "--data-column",
        type=str,
        default="DATA",
        help="Column containing the data to calibrate. ",
    )
    band_parser.add_argument(
        "--mode",
        type=str,
        default="calibrate",
        choices=supported_models,
        help=f"Support 1934-638 calibration models. available models: {supported_models}. ",
    )
    band_parser.add_argument(
        "--container",
        type=Path,
        default=Path("calibrate.sif"),
        help="Path to container that is capable or running and apply calibration solutions for desired mode. ",
    )

    plot_parser = subparser.add_parser("plot", help="Plot a previous bandpass solution")

    plot_parser.add_argument(
        "solutions", type=Path, help="Path to the AO-style solutions file to plot"
    )
    plot_parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Path to write plot to. If None, it is automatically assigned based on solutions name. ",
    )

    return parser


def cli() -> None:
    import logging

    logger.setLevel(logging.INFO)

    parser = get_parser()

    args = parser.parse_args()

    if args.mode == "calibrate":
        calibrate_bandpass(
            ms_path=args.ms,
            data_column=args.data_column,
            mode=args.mode,
            container=args.container,
        )
    elif args.mode == "plot":
        plot_solutions(solutions_path=args.solutions)
    else:
        logger.warn("This should not have happened. ")


if __name__ == "__main__":
    cli()
