"""Procedure to calibrate bandpass observation
"""
from argparse import ArgumentParser
from pathlib import Path
from typing import Union, Optional

import numpy as np
from casacore.tables import table, taql

from flint.logging import logger
from flint.ms import MS, describe_ms
from flint.sky_model import KNOWN_1934_FILES, get_1934_model
from flint.calibrate.aocalibrate import calibrate_apply_ms, AOSolutions
from flint.flagging import flag_ms_aoflagger


def plot_solutions(solutions_path: Path, ref_ant: int = 0) -> None:
    """Plot solutions for AO-style solutions

    Args:
        solutions_path (Path): Path to the solutions file
        ref_ant (int, optional): Reference antenna to use. Defaults to 0.
    """
    logger.info(f"Plotting {solutions_path}")

    ao_sols = AOSolutions.load(path=solutions_path)
    plot_paths = ao_sols.plot_solutions(ref_ant=ref_ant)


def flag_bandpass_offset_pointings(ms: Union[MS, Path]) -> MS:
    """The typical bandpass style observation in ASKAP will shift each beam
    so that it is centred on the bandpass-calibration object (here B1934-638).
    During each offset position all beams are recording data still. The trick
    here is that all 36 fields are still recorded in the measurement set:

    fields: ['B1934-638_beam0', 'B1934-638_beam1', 'B1934-638_beam10', 'B1934-638_beam11',
    'B1934-638_beam12', 'B1934-638_beam13', 'B1934-638_beam14', 'B1934-638_beam15',
    'B1934-638_beam16', 'B1934-638_beam17', 'B1934-638_beam18', 'B1934-638_beam19',
    'B1934-638_beam2', 'B1934-638_beam20', 'B1934-638_beam21', 'B1934-638_beam22',
    'B1934-638_beam23', 'B1934-638_beam24', 'B1934-638_beam25', 'B1934-638_beam26',
    'B1934-638_beam27', 'B1934-638_beam28', 'B1934-638_beam29', 'B1934-638_beam3',
    'B1934-638_beam30', 'B1934-638_beam31', 'B1934-638_beam32', 'B1934-638_beam33',
    'B1934-638_beam34', 'B1934-638_beam35', 'B1934-638_beam4', 'B1934-638_beam5',
    'B1934-638_beam6', 'B1934-638_beam7', 'B1934-638_beam8', 'B1934-638_beam9']

    This function will attempt to deduce the intended field name for the beam
    in question, and then flag all other fields.

    Args:
        ms (Union[MS, Path]): Path or instance of MS describing the measurement set to flag all other bandpass field.

    Returns:
        MS: A description of the ms
    """
    ms = MS(path=ms) if isinstance(ms, Path) else ms
    ms_summary = describe_ms(ms, verbose=False)

    good_field_name = f"B1934-638_beam{ms_summary.beam}"
    logger.info(f"The B1934-638 field name is {good_field_name}. ")
    logger.info(f"Will attempt to flag other fields. ")

    with table(f"{str(ms.path)}/FIELD", readonly=True, ack=False) as tab:
        # The ID is _position_ of the matching row in the table.
        field_names = tab.getcol("NAME")
        field_idx = np.argwhere([fn == good_field_name for fn in field_names])[0]

        assert (
            len(field_idx) == 1
        ), f"More than one matching field name found. This should not happen. {good_field_name=} {field_names=}"

        field_idx = field_idx[0]
        logger.info(f"{good_field_name} FIELD_ID is {field_idx}")

    with table(f"{str(ms.path)}", readonly=False, ack=False) as tab:
        field_idxs = tab.getcol("FIELD_ID")
        field_mask = field_idxs != field_idx
        logger.info(
            f"Found {np.sum(field_mask)} rows not matching FIELD_ID={field_idx}"
        )

        # This is asserting that the stored polarisations are all XX, XY, YX, YY
        flag_row = np.array([True, True, True, True])
        flags = tab.getcol("FLAG")
        flags[field_mask] = flag_row

        # Update the column again
        tab.putcol("FLAG", flags)

    return ms


def extract_correct_bandpass_pointing(
    ms: Union[MS, Path], source_name_prefix: str = "B1934-638"
) -> MS:
    """The typical bandpass style observation in ASKAP will shift each beam
    so that it is centred on the bandpass-calibration object (here B1934-638).
    During each offset position all beams are recording data still. The trick
    here is that all 36 fields are still recorded in the measurement set:

    fields: ['B1934-638_beam0', 'B1934-638_beam1', 'B1934-638_beam10', 'B1934-638_beam11',
    'B1934-638_beam12', 'B1934-638_beam13', 'B1934-638_beam14', 'B1934-638_beam15',
    'B1934-638_beam16', 'B1934-638_beam17', 'B1934-638_beam18', 'B1934-638_beam19',
    'B1934-638_beam2', 'B1934-638_beam20', 'B1934-638_beam21', 'B1934-638_beam22',
    'B1934-638_beam23', 'B1934-638_beam24', 'B1934-638_beam25', 'B1934-638_beam26',
    'B1934-638_beam27', 'B1934-638_beam28', 'B1934-638_beam29', 'B1934-638_beam3',
    'B1934-638_beam30', 'B1934-638_beam31', 'B1934-638_beam32', 'B1934-638_beam33',
    'B1934-638_beam34', 'B1934-638_beam35', 'B1934-638_beam4', 'B1934-638_beam5',
    'B1934-638_beam6', 'B1934-638_beam7', 'B1934-638_beam8', 'B1934-638_beam9']

    This function will attempt to deduce the intended field name for the beam
    in question, and then create a new measurement set with just these data. It
    internally uses `taql` to create a selection statement:

    >>> with table(ms_path) as tab:
    >>>    sub_ms = taql("select * from $tab where FIELD_ID==field_idx")
    >>>    sub_ms.copy(out_path, deep=True)

    Therefore, some properties from other tables (e.g. FIELDS) may still
    contain references to other fields.

    Args:
        ms (Union[MS, Path]): Path or instance of MS describing the measurement set to flag all other bandpass field.
        source_name_prefix (str, optional): The beginning of the source name stored in the NAME column of the FIELD table.
        Field names are of the form B1934-638_beam1, where B1934-638 would be the prefix name, and beam is constructed based
        on the beam among the 36 observing the target source (for example).

    Returns:
        MS: A description of the new measurement set created with the file name
        ending .beamN.ms.
    """
    ms = MS(path=ms) if isinstance(ms, Path) else ms
    ms_summary = describe_ms(ms, verbose=False)

    good_field_name = f"{source_name_prefix}_beam{ms_summary.beam}"
    field_id = ms.get_field_id_for_field(field_name=good_field_name)

    out_path = ms.path.with_suffix(f".{source_name_prefix}.beam{ms_summary.beam}.ms")
    logger.info(f"Will create a MS, writing to {out_path}")

    with table(f"{str(ms.path)}") as tab:
        field_ms = taql(f"select * from $tab where FIELD_ID=={field_id}")
        field_ms.copy(str(out_path), deep=True)

    return ms.with_options(path=out_path)


def calibrate_bandpass(
    ms_path: Path,
    data_column: str,
    mode: str,
    calibrate_container: Path,
    plot: bool = True,
    aoflagger_container: Optional[Path] = None,
) -> MS:
    """Entry point to extract the appropriate field from a bandpass observation,
    run AO-style calibrate, and plot results. In its current form a new measurement
    set will be created container just the appropriate field to calibrate.

    Args:
        ms_path (Path): Path the the measurement set containing bandpass obervations of B1934-638
        data_column (str): The column that will be calibrated.
        mode (str): The calibration approach to use. Currently only `calibrate` is supported.
        calibrate_container (Path): The path to the singularity container that holds the appropriate software.
        plot (bool, optional): Whether plotting should be performed. Defaults to True.
        aoflagger_container (Path): The path to the singularity container that holds aoflagger.
        If this is not `None` that flagging will be performed on the extracted field-specific measurement
        set.

    Returns:
        MS: The calibrated measurement set with nominated column
    """
    logger.info(f"Will calibrate {str(ms_path)}, colum {data_column}")

    # TODO: Check to make sure only 1934-638
    model_path: Path = get_1934_model(mode=mode)

    ms = extract_correct_bandpass_pointing(ms=ms_path)
    describe_ms(ms=ms)

    if aoflagger_container is not None:
        logger.info(f"Will run flagger on extracted measurement set. ")
        flag_ms_aoflagger(
            ms=ms.with_options(column="DATA"), container=aoflagger_container
        )

    apply_solutions = calibrate_apply_ms(
        ms_path=ms.path,
        model_path=model_path,
        container=calibrate_container,
        data_column=data_column,
    )

    if plot:
        plot_solutions(solutions_path=apply_solutions.solution_path)

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
        "--calibrate-container",
        type=Path,
        default=Path("calibrate.sif"),
        help="Path to container that is capable or running and apply calibration solutions for desired mode. ",
    )
    band_parser.add_argument(
        "--aoflagger-container",
        type=Path,
        default=None,
        help="Path to container container aoflagger. If provided guided automated flagging with aoflagger will be performed on the extracted measurement set.  ",
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
            calibrate_container=args.calibrate_container,
            aoflagger_container=args.aoflagger_container,
        )
    elif args.mode == "plot":
        plot_solutions(solutions_path=args.solutions)
    else:
        logger.warn("This should not have happened. ")


if __name__ == "__main__":
    cli()
