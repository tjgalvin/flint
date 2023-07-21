"""Utilities related to using casa tasks to perform self-calibration.

This tooling is mostly centred on using gaincal from casatasks.
"""
from __future__ import annotations
import regex
from typing import Optional, NamedTuple, Dict
from shutil import copytree, rmtree
from argparse import ArgumentParser
from pathlib import Path

from casatasks import gaincal, applycal
from casacore.tables import table

from flint.logging import logger
from flint.ms import MS
from flint.flagging import nan_zero_extreme_flag_ms


class GainCalOptions(NamedTuple):
    """Options provided to the casatasks gaincal function. Most options correspond to those in gaincal."""

    solint: str = "60s"
    """Solution length interval"""
    calmode: str = "p"
    """Calibration mode. """
    round: int = 0
    """Self-calibration round. Not a gaincal option. """
    minsnr: float = 0.0
    """Minimum signal-to-noise of the solutions. Below this the solutions and data are flagged. """
    uvrange: str = ">200m"
    """Data selected to go through calibration procedure"""
    selectdata: bool = True
    """Whether data selection actions will be applied in gaincal. """
    gaintype: str = "G"
    """The gain type that would be solved for. """

    def with_options(self, **kwargs) -> GainCalOptions:
        _dict = self._asdict()
        _dict.update(**kwargs)

        return GainCalOptions(**_dict)


def get_selfcal_ms_name(in_ms_path: Path, round: int = 1) -> Path:
    """Create the new output MS path that will be used for self-calibration. The
    output measurement set path will include a roundN.ms suffix, where N is the
    round. If such a suffic already exists from an earlier self-calibration round,
    it will be removed and replaced.

    Args:
        in_ms_path (Path): The measurement set that will go through self-calibration
        round (int, optional): The self-calibration round number that is currently being used. Defaults to 1.

    Returns:
        Path: Output measurement set path to use
    """
    res = regex.search("\\.round[0-9]+.ms", str(in_ms_path.name))
    if res:
        logger.info(f"Detected a previous round of self-calibration. ")
        span = res.span()
        name_str = str(in_ms_path.name)
        name = f"{name_str[:span[0]]}.round{round}.ms"
    else:
        name = f"{str(in_ms_path.stem)}.round{round}.ms"
    out_ms_path = in_ms_path.parent / name

    assert (
        in_ms_path != out_ms_path
    ), f"{in_ms_path=} and {out_ms_path=} match. Something went wrong when creating new self-cal name. "

    return out_ms_path


def copy_and_clean_ms_casagain(ms: MS, round: int = 1) -> MS:
    """Create a copy of a measurement set in preparation for selfcalibration
    using casa's gaincal and applycal. Applycal only works when calibrating
    DATA and creating a CORRECTED_DATA column. Columns are removed in the
    copied MS to allow this.

    Args:
        ms (MS): Measurement set that would go through self-calibration.
        round (int, optional): The self-calibration round. Defaults to 1.

    Returns:
        MS: Copy of input measurement set with columns removed as required.
    """
    # TODO: Excellent function tto start to get the test framework working from!
    # TODO: Update this name creating to a single location
    out_ms_path = get_selfcal_ms_name(in_ms_path=ms.path, round=round)

    logger.info(f"Output MS name will be {str(out_ms_path)}.")
    logger.info(f"Copying {ms.path} to {out_ms_path}.")

    if out_ms_path.exists():
        logger.warn(f"{out_ms_path} already exists. Removing it. ")
        rmtree(out_ms_path)

    copytree(ms.path, out_ms_path)

    logger.info(f"Copying finished. ")

    # The casa gaincal and applycal tasks __really__ expect the input and output
    # column names to be DATA and CORRECTED_DATA. So, here we will go through the
    # motions of deleting and rnaming columns. Note that the MODEL_DATA column needs
    # to exist. The INSTRUMENT_DATA column will also be removed.
    logger.info("About to open the table. ")
    with table(str(out_ms_path), readonly=False, ack=False) as tab:
        logger.info("About tto get the colnames")
        colnames = tab.colnames()
        logger.info(f"Column names are: {colnames}")
        to_delete = ["DATA", "INSTRUMENT_DATA"]
        for col in to_delete:
            if col in colnames:
                logger.info(f"Removing {col=} from {str(out_ms_path)}.")
                try:
                    tab.removecols(col)
                except:
                    logger.critical(f"Failed to remove {col=}!")
            else:
                logger.warn(f"Column {col} not found in {str(out_ms_path)}.")

        logger.info(f"Renaming CORRECTED_DATA to DATA. ")
        tab.renamecol("CORRECTED_DATA", "DATA")

    ms = ms.with_options(path=out_ms_path, column="DATA")

    ms = nan_zero_extreme_flag_ms(ms=ms)

    return ms


def gaincal_applycal_ms(
    ms: MS,
    round: int = 1,
    gain_cal_options: Optional[GainCalOptions] = None,
    update_gain_cal_options: Optional[Dict[str, Any]] = None,
) -> MS:
    """Perform self-calibration using casa's gaincal and applycal tasks against
    an input measurement set.

    Args:
        ms (MS): Measurement set that will be self-calibrated.
        round (int, optional): Round of self-calibration, which is used for unique names. Defaults to 1.
        gain_cal_options (Optional[GainCalOptions], optional): Options provided to gaincal. Defaults to None.
        update_gain_cal_options (Optional[Dict[str, Any]], optional): Update the gain_cal_options with these. Defaults to None.

    Returns:
        MS: _description_
    """
    logger.info(f"Inputer ms={ms}")

    if gain_cal_options is None:
        gain_cal_options = GainCalOptions()
    if update_gain_cal_options:
        logger.info(f"Updating gaincal options with: {update_gain_cal_options}")
        gain_cal_options = gain_cal_options.with_options(**update_gain_cal_options)

    cal_ms = copy_and_clean_ms_casagain(ms=ms, round=round)

    cal_table = cal_ms.path.absolute().parent / cal_ms.path.with_suffix(f".caltable")
    logger.info(f"Will create calibration table {cal_table}.")

    if cal_table.exists():
        logger.warn(f"Removing {str(cal_table)}")
        rmtree(cal_table)

    gaincal(
        vis=str(cal_ms.path),
        caltable=str(cal_table),
        solint=gain_cal_options.solint,
        gaintype=gain_cal_options.gaintype,
        minsnr=gain_cal_options.minsnr,
        calmode=gain_cal_options.calmode,
        selectdata=gain_cal_options.selectdata,
        uvrange=gain_cal_options.uvrange,
    )

    if not cal_table.exists():
        logger.critical(
            f"The calibration table was not created. Likely gaincal failed. "
        )
        return ms

    logger.info("Solutions have been solved. Applying them. ")

    applycal(vis=str(cal_ms.path), gaintable=str(cal_table))

    return cal_ms.with_options(column="CORRECTED_DATA")


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)

    sub_parsers = parser.add_subparsers(dest="mode")

    gaincal_parser = sub_parsers.add_parser("gaincal")
    gaincal_parser.add_argument(
        "ms", type=Path, help="Path to the measurement set to calibrate. "
    )
    gaincal_parser.add_argument(
        "--round", type=int, default=1, help="Self-calibration round number. "
    )

    return parser


def cli() -> None:
    parser = get_parser()

    args = parser.parse_args()

    if args.mode == "gaincal":
        gaincal_applycal_ms(ms=MS(path=args.ms), round=args.round)


if __name__ == "__main__":
    cli()
