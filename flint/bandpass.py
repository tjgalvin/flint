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
    logger.info(f"Plotting {solutions_path}")

    ao_sols = AOSolutions.load(path=solutions_path)

    if ao_sols.nsol > 1:
        logger.warn(f"Found {ao_sols.nsol} intervals, plotting the first. ")
    plot_sol = 0  # The first time interval

    data = ao_sols.bandpass[plot_sol] / ao_sols.bandpass[plot_sol, ref_ant, :, :]
    amplitudes = np.abs(data)
    phases = np.angle(data, deg=True)
    channels = np.arange(ao_sols.nchan)

    ncolumns = 6
    nrows = ao_sols.nant // ncolumns
    if ncolumns * nrows < ao_sols.nant:
        nrows += 1
    logger.debug(f"Plotting {plot_sol=} with {ncolumns=} {nrows=}")

    fig_amp, axes_amp = plt.subplots(nrows, ncolumns, figsize=(14, 14))
    fig_phase, axes_phase = plt.subplots(nrows, ncolumns, figsize=(14, 14))

    for y in range(nrows):
        for x in range(ncolumns):
            ant = y * nrows + x

            amps_xx = amplitudes[ant, :, 0]
            amps_yy = amplitudes[ant, :, 3]
            phase_xx = phases[ant, :, 0]
            phase_yy = phases[ant, :, 3]

            if np.sum(~np.isfinite([amps_xx, amps_yy])) == 0:
                logger.warn(f"No valid data for {ant=}")
                continue

            max_amp_xx = np.nanmax(amps_xx[np.isfinite(amps_xx)])
            max_amp_yy = np.nanmax(amps_yy[np.isfinite(amps_yy)])
            max_amp = np.max([max_amp_xx, max_amp_yy])
            ax_a, ax_p = axes_amp[y, x], axes_phase[y, x]
            ax_a.plot(channels, amps_xx, marker=None, color="black")
            ax_a.plot(channels, amps_yy, marker=None, color="red")
            ax_a.set(ylim=(0, 1.2 * max_amp))
            ax_a.set_title(f"ak{ant:02d}", fontsize=8)

            ax_p.plot(channels, phase_xx, marker=None, color="black")
            ax_p.plot(channels, phase_yy, marker=None, color="red")
            ax_p.set(ylim=(-200, 200))
            ax_p.set_title(f"ak{ant:02d}", fontsize=8)

    fig_amp.suptitle(f"Amplitudes")
    fig_phase.suptitle(f"Phases")

    fig_amp.tight_layout()
    fig_phase.tight_layout()

    out_amp = f"{str(solutions_path.with_suffix('.amplitude.pdf'))}"
    logger.info(f"Saving {out_amp}.")
    fig_amp.savefig(out_amp)

    out_phase = f"{str(solutions_path.with_suffix('.phase.pdf'))}"
    logger.info(f"Saving {out_phase}.")
    fig_phase.savefig(out_phase)


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
