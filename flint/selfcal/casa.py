"""Utilities related to using casa tasks to perform self-calibration.

This tooling is mostly centred on using gaincal from casatasks.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from shutil import copytree
from typing import Any, Dict, NamedTuple, Optional

import regex
from casacore.tables import table
from casatasks import applycal, cvel, gaincal, mstransform

from flint.exceptions import GainCalError
from flint.flagging import nan_zero_extreme_flag_ms
from flint.logging import logger
from flint.ms import MS
from flint.utils import remove_files_folders, rsync_copy_directory, zip_folder


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
    nspw: int = 1
    """The number of spectral windows to use during the self-calibration rountine. If 1, no changes
    are made to the measurement set. If more than 1, then the measurement will be reformed to form
    a new measurement set conforming to the number of spws set. This process can be fragile as the
    casa tasks sometimes do not like the configurate, so ye warned."""

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
        logger.info("Detected a previous round of self-calibration. ")
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


def copy_and_clean_ms_casagain(ms: MS, round: int = 1, verify: bool = True) -> MS:
    """Create a copy of a measurement set in preparation for selfcalibration
    using casa's gaincal and applycal. Applycal only works when calibrating
    DATA and creating a CORRECTED_DATA column. Columns are removed in the
    copied MS to allow this.

    Args:
        ms (MS): Measurement set that would go through self-calibration.
        round (int, optional): The self-calibration round. Defaults to 1.
        verify (bool, optional): Verify that copying the measurementt set (done in preparation for gaincal) was successful. Uses a call to rsync. Defaults to True.

    Returns:
        MS: Copy of input measurement set with columns removed as required.
    """
    # TODO: Excellent function tto start to get the test framework working from!
    # On the above: No idea what this pirate is talking about
    # TODO: Update this name creating to a single location
    out_ms_path = get_selfcal_ms_name(in_ms_path=ms.path, round=round)

    logger.info(f"Output MS name will be {str(out_ms_path)}.")
    logger.info(f"Copying {ms.path} to {out_ms_path}.")

    if out_ms_path.exists():
        logger.warning(f"{out_ms_path} already exists. Removing it. ")
        remove_files_folders(out_ms_path)

    copytree(ms.path, out_ms_path)
    # Because we can trust nothing, verify the
    # copy with rsync. On some lustre file systems (mostly seen on stonix)
    # components of the measurement sett are not always successfully
    # copied with a simple copytree.
    if verify:
        rsync_copy_directory(ms.path, out_ms_path)

    logger.info("Copying finished. ")

    # The casa gaincal and applycal tasks __really__ expect the input and output
    # column names to be DATA and CORRECTED_DATA. So, here we will go through the
    # motions of deleting and rnaming columns. Note that the MODEL_DATA column needs
    # to exist. The INSTRUMENT_DATA column will also be removed.
    logger.info("About to open the table. ")
    with table(str(out_ms_path), readonly=False, ack=False) as tab:
        logger.info("About tto get the colnames")
        colnames = tab.colnames()
        logger.info(f"Column names are: {colnames}")
        if ms.column == "DATA" and "CORRECTED_DATA" not in colnames:
            logger.info("Data is the nominated column, and CORRECTED_DATA does not exist. Returning. ")
        else:
            to_delete = ["DATA", ]
            for col in to_delete:
                if col in colnames:
                    logger.info(f"Removing {col=} from {str(out_ms_path)}.")
                    try:
                        tab.removecols(col)
                        tab.flush(recursive=True)
                    except Exception as e:
                        logger.critical(f"Failed to remove {col=}! \nCaptured error: {e}")
                else:
                    logger.warning(f"Column {col} not found in {str(out_ms_path)}.")

            logger.info("Renaming CORRECTED_DATA to DATA. ")
            tab.renamecol("CORRECTED_DATA", "DATA")

            # If we have removed original DATA and renamed the CORRECTED_DATA, we need to update
            # the nominated column (which was previously the CORRECT_DATA and now renamed DATA),
            # This pirate got confused
            ms = ms.with_options(path=out_ms_path, column="DATA")

    ms = nan_zero_extreme_flag_ms(ms=ms)

    return ms


def create_spws_in_ms(ms_path: Path, nspw: int) -> Path:
    """Use the casa task mstransform to create `nspw` spectral windows
    in the input measurement set. This is necessary when attempting to
    use gaincal to solve for some frequency-dependent solution.

    Args:
        ms_path (Path): The measurement set that should be reformed to have `nspw` spectral windows
        nspw (int): The number of spectral windows to create

    Returns:
        Path: The path to the measurement set that was updated
    """

    logger.info(f"Transforming {str(ms_path)} to have {nspw} SPWs")
    transform_ms = ms_path.with_suffix(".ms_transform")

    mstransform(
        vis=str(ms_path),
        outputvis=str(transform_ms),
        regridms=True,
        nspw=nspw,
        mode="channel",
        nchan=-1,
        start=0,
        width=1,
        chanbin=1,
        createmms=False,
        datacolumn="all",
        combinespws=False,
    )

    logger.info(
        f"Successfully created the transformed measurement set {transform_ms} with {nspw} SPWs."
    )
    remove_files_folders(ms_path)

    logger.info(f"Renaming {transform_ms} to {ms_path}.")
    transform_ms.rename(ms_path)

    # Look above - we have renamed the cvel measurement set Captain
    return ms_path


def merge_spws_in_ms(ms_path: Path) -> Path:
    """Attempt to merge together all SPWs in the input measurement
    set using the `cvel` casa task. This can be a little fragile.

    The `cvel` task creates a new measurement set, so there would
    temporially be a secondary measurment set.

    Args:
        ms_path (Path): The measurement set that should have its SPWs merged together.

    Returns:
        Path: The measurement set with merged SPWs. It will have the same path as `ms_path`, but is a new measurement set.
    """
    logger.info("Will attempt to merge all spws using cvel.")

    cvel_ms_path = ms_path.with_suffix(".cvel")
    cvel(vis=str(ms_path), outputvis=str(cvel_ms_path), mode="channel_b")

    logger.info(f"Successfully merged spws in {cvel_ms_path}")

    remove_files_folders(ms_path)

    logger.info(f"Renaming {cvel_ms_path} to {ms_path}")
    cvel_ms_path.rename(ms_path)

    # Look above - we have renamed the cvel measurement set Captain
    return ms_path


def gaincal_applycal_ms(
    ms: MS,
    round: int = 1,
    gain_cal_options: Optional[GainCalOptions] = None,
    update_gain_cal_options: Optional[Dict[str, Any]] = None,
    archive_input_ms: bool = False,
    raise_error_on_fail: bool = True,
) -> MS:
    """Perform self-calibration using casa's gaincal and applycal tasks against
    an input measurement set.

    Args:
        ms (MS): Measurement set that will be self-calibrated.
        round (int, optional): Round of self-calibration, which is used for unique names. Defaults to 1.
        gain_cal_options (Optional[GainCalOptions], optional): Options provided to gaincal. Defaults to None.
        update_gain_cal_options (Optional[Dict[str, Any]], optional): Update the gain_cal_options with these. Defaults to None.
        archive_input_ms (bool, optional): If True, the input measurement set will be compressed into a single file. Defaults to False.
        raise_error_on_fail (bool, optional): If gaincal does not converge raise en error. if False and gain cal fails return the input ms. Defaults to True.

    Raises:
        GainCallError: Raised when raise_error_on_fail is True and gaincal does not converge.

    Returns:
        MS: THe self-calibrated measurement set.
    """
    logger.info(f"Measurement set to be self-calibrated: ms={ms}")

    if gain_cal_options is None:
        gain_cal_options = GainCalOptions()
    if update_gain_cal_options:
        logger.info(f"Updating gaincal options with: {update_gain_cal_options}")
        gain_cal_options = gain_cal_options.with_options(**update_gain_cal_options)

    cal_ms = copy_and_clean_ms_casagain(ms=ms, round=round)

    cal_table = cal_ms.path.absolute().parent / cal_ms.path.with_suffix(".caltable")
    logger.info(f"Will create calibration table {cal_table}.")

    if cal_table.exists():
        logger.warning(f"Removing {str(cal_table)}")
        remove_files_folders(cal_table)

    # This is used for when a frequency dependent self-calibration solution is requested.
    # Apparently in the casa way of life the gaincal task (used below) automatically does
    # this when it detects multiple spectral windows in the measurement set. By default,
    # the typical ASKAP MS has a single one. When nspw > 1, a combination of mstransorm+cvel
    # is used to add multiple spws, gaincal, applycal, cvel back to a single spw. Why a
    # single spw? Some tasks just work better with it - and this pirate likes a simple life
    # on the seven seas. Also have no feeling of what the yandasoft suite prefers.
    if gain_cal_options.nspw > 1:
        cal_path = create_spws_in_ms(ms_path=cal_ms.path, nspw=gain_cal_options.nspw)
        # At the time of writing the output path returned above should always
        # be the same as the ms_path=, however me be a ye paranoid pirate who
        # trusts no one of the high seas
        cal_ms = cal_ms.with_options(path=cal_path)

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
            "The calibration table was not created. Likely gaincal failed. "
        )
        if raise_error_on_fail:
            raise GainCalError(f"Gaincal failed for {cal_ms.path}")
        else:
            return ms

    logger.info("Solutions have been solved. Applying them. ")

    applycal(vis=str(cal_ms.path), gaintable=str(cal_table))

    # This is used for when a frequency dependent self-calibration solution is requested
    # It is often useful (mandatory!) to have a single spw for some tasks - both of the casa
    # and everyone else varity.
    if gain_cal_options.nspw > 1:
        # putting it all back to a single spw
        cal_ms_path = merge_spws_in_ms(ms_path=cal_ms.path)
        # At the time of writing merge_spws_in_ms returns the ms_path=,
        # but this pirate trusts no one.
        cal_ms = cal_ms.with_options(path=cal_ms_path)

    if archive_input_ms:
        zip_folder(in_path=ms.path)

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
