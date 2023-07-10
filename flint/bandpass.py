"""Procedure to calibrate bandpass observation
"""
from argparse import ArgumentParser
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from casacore.tables import table

from flint.logging import logger
from flint.ms import MS, describe_ms
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


def flag_offset_pointings(ms: Union[MS, Path]) -> None:
    ms = MS(path=ms) if isinstance(ms, Path) else ms
    ms_summary = describe_ms(ms)

    good_field_name = f"B1934-638_beam{ms_summary.beam}"
    logger.info(f"The B1934-638 field name is {good_field_name}. ")
    logger.info(f"Will attempt to flag other fields. ")

    with table(f"{str(ms.path)}/FIELD", readonly=True) as tab:
        # The ID is _position_ of the matching row in the table.
        field_names = tab.getcol("NAME")
        field_idx = np.argwhere([fn == good_field_name for fn in field_names])[0]

        assert (
            len(field_idx) == 1
        ), f"More than one matching field name found. This should not happen. {good_field_name=} {field_names=}"

        field_idx = field_idx[0]
        logger.info(f"{good_field_name} FIELD_ID is {field_idx}")

    with table(f"{str(ms.path)}") as tab:
        field_idxs = tab.getcol("FIELD_ID")
        field_mask = field_idxs != field_idx
        logger.info(
            f"Found {np.sum(field_mask)} rows not matching FIELD_ID={field_idx}"
        )

        # This is asserting that the stored polarisations are all XX, XY, YX, YY
        flag_row = np.array([True, True, True, True])
        flags = tab.getcol("FLAG")
        flags[field_mask] = flag_row


def calibrate_bandpass(
    ms_path: Path, data_column: str, mode: str, container: Path
) -> MS:
    logger.info(f"Will calibrate {str(ms_path)}, colum {data_column}")

    # TODO: Check to make sure only 1934-638
    model_path: Path = get_1934_model(mode=mode)

    describe_ms(ms=ms_path)
    flag_offset_pointings(ms=ms_path)

    # ms = calibrate_apply_ms(
    #     ms_path=ms_path,
    #     model_path=model_path,
    #     container=container,
    #     data_column=data_column,
    # )

    return


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
