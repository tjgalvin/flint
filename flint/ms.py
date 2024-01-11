"""Operations related to measurement sets
"""
from __future__ import (  # Used for mypy/pylance to like the return type of MS.with_options
    annotations,
)

from argparse import ArgumentParser
from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from shutil import rmtree
from typing import List, NamedTuple, Optional, Union

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time
from casacore.tables import table, taql
from fixms.fix_ms_corrs import fix_ms_corrs
from fixms.fix_ms_dir import fix_ms_dir

from flint.logging import logger
from flint.naming import create_ms_name
from flint.utils import rsync_copy_directory


class MS(NamedTuple):
    """Helper to keep tracked of measurement set information"""

    path: Path
    """Path to the measurement set that is being represented"""
    column: Optional[str] = None
    """Column that should be operated against"""
    beam: Optional[int] = None
    """The beam ID of the MS within an ASKAP field"""
    field: Optional[str] = None
    """The field name  of the data"""
    model_column: Optional[str] = None
    """The column name of the most recently MODEL data"""

    def get_field_id_for_field(self, field_name: str) -> Union[int, None]:
        """Return the FIELD_ID for an elected field in a measurement set. See
        `flink.ms.get_field_id_for_field` for full details.
        """

        return get_field_id_for_field(ms=self, field_name=field_name)

    @classmethod
    def cast(cls, ms: Union[MS, Path]) -> MS:
        """Create a MS instance, if necessary, given eith a Path or MS.

        Args:
            ms (Union[MS, Path]): The input type to consider

        Returns:
            MS: A normalised MS
        """
        ms = ms if isinstance(ms, MS) else MS(path=ms)

        return ms

    def with_options(self, **kwargs) -> MS:
        """Create a new MS instance with keywords updated

        Returns:
            MS: New MS instance with updated attributes
        """
        # TODO: Update the signature to have the actual attributes to
        # help keep mypy and other linters happy
        as_dict = self._asdict()
        as_dict.update(kwargs)

        return MS(**as_dict)


class MSSummary(NamedTuple):
    """Small structure to contain overview of a MS"""

    unflagged: int
    """Number of unflagged records"""
    flagged: int
    """Number of flagged records"""
    fields: List[str]
    """Collection of unique field names from the FIELDS table"""
    ants: List[int]
    """Collection of unique antennas"""
    beam: int
    """The ASKAP beam number of the measurement set"""


# TODO: Some common MS validation functions?
# - list / number of fields
# - new name function (using names / beams)
# - check to see if fix_ms_dir / fix_ms_corrs
# - delete column/rename column


@contextmanager
def critical_ms_interaction(
    input_ms: Path, copy: bool = False, suffix: str = ".critical"
):
    """A context manager that can be used to register a measurement set as
    entering a critical segment, i.e. phase rotation. If this stage were to
    fail, then future operations against said measurement set might be
    nonsense. This mechanism is intended to make it clear that the measurement
    set is in a dangerous part of code.

    Failure to return the MS to its orignal name (or rename the copy) highlights
    this failed stage.

    Args:
        input_ms (Path): The measurement set to monitor.
        copy (bool, optional): If True, a copy of the MS is made with the suffix supplied. Otherwise, the MS is siomply renamed. Defaults to False.
        suffix (str, optional): Suffix indicating the MS is in the dangerous stage. Defaults to '.critical'.

    Yields:
        Path: Resource location of the measurement set being processed
    """
    input_ms = Path(input_ms)
    output_ms: Path = Path(input_ms.with_suffix(suffix))

    # Make sure that the measurement set to preserve exists, and there is not already
    # a target output measurement set already on disk. This second check would be
    # useful in copy==True mode, ya seadog
    assert input_ms.exists(), f"The input measurement set {input_ms} does not exist. "
    assert (
        not output_ms.exists()
    ), f"The output measurement set {output_ms} already exists. "

    if copy:
        rsync_copy_directory(target_path=input_ms, out_path=output_ms)
    else:
        input_ms.rename(target=output_ms)

    try:
        yield output_ms
    except Exception as e:
        logger.error(
            f"An error occured when interacting with {input_ms} during a critical stage. "
        )
        raise e

    # If we get to here, things worked successfully, and we
    # should return things back to normal.
    if copy:
        rmtree(input_ms)
        output_ms.rename(input_ms)
    else:
        output_ms.rename(target=input_ms)


def get_field_id_for_field(ms: Union[MS, Path], field_name: str) -> Union[int, None]:
    """Return the FIELD_ID for an elected field in a measurement set

    Args:
        ms (Union[MS, Path]): The measurement set to inspect
        field_name (str): The desired name of the field to find the FIELD_ID for

    Raises:
        ValueError: Raised when more than one unique FIELD_ID is found for a single field

    Returns:
        Union[int, None]: The FIELD_ID as an `int` if the field is found. `None` if
        it is not found.
    """
    ms_path = ms if isinstance(ms, Path) else ms.path

    with table(f"{str(ms_path)}/FIELD", readonly=True, ack=False) as tab:
        # The ID is _position_ of the matching row in the table.
        field_names = tab.getcol("NAME")
        field_idx = np.argwhere([fn == field_name for fn in field_names])[0]

        if len(field_idx) == 0:
            return None
        elif len(field_idx) > 1:
            raise ValueError(
                f"More than one matching field name found. This should not happen. {field_name=} {field_names=}"
            )

        field_idx = field_idx[0]
        logger.info(f"{field_name} FIELD_ID is {field_idx}")

    return field_idx


def get_beam_from_ms(ms: Union[MS, Path]) -> int:
    """Lookup the ASKAP beam number from a measurement set.

    Args:
        ms (Union[MS, Path]): The measurement set to inspect. If `MS`, the attached path is used.

    Returns:
        int: The beam ID number
    """
    ms_path = ms if isinstance(ms, Path) else ms.path

    with table(str(ms_path), readonly=True, ack=False) as tab:
        uniq_beams = sorted(list(set(tab.getcol("FEED1"))))

    assert (
        len(uniq_beams) == 1
    ), f"Expected {str(ms_path)} to contain a single beam, found {len(uniq_beams)}: {uniq_beams=}"

    return uniq_beams[0]


def get_freqs_from_ms(ms: Union[MS, Path]) -> np.ndarray:
    """Return the frequencies observed from an ASKAP Meaurement set.
    Some basic checks are performed to ensure they conform to some
    expectations.

    Args:
        ms (Union[MS, Path]): Measurement set to inspect

    Returns:
        np.ndarray: A squeeze array of frequencies, in Hertz.
    """
    ms = MS.cast(ms)

    with table(f"{str(ms.path)}/SPECTRAL_WINDOW", readonly=True, ack=False) as tab:
        freqs = tab.getcol("CHAN_FREQ")

    freqs = np.squeeze(freqs)
    assert (
        len(freqs.shape) == 1
    ), f"Frequency axis has dimensionality greater than one. Not expecting that. {len(freqs.shape)}"

    return freqs


def get_times_from_ms(ms: Union[MS, Path]) -> Time:
    """Return the observation times from an ASKAP Measurement set.

    Args:
        ms (Union[MS, Path]): Measurement set to inspect

    Returns:
        Time: The observation times
    """
    # Get the time of the observation
    ms = MS.cast(ms)
    with table(str(ms.mpath), ack=False) as tab:
        times = Time(tab.getcol("TIME") * u.s, format="mjd")

    return times


def get_telescope_location_from_ms(ms: Union[MS, Path]) -> EarthLocation:
    """Return the telescope location from an ASKAP Measurement set.

    Args:
        ms (Union[MS, Path]): Measurement set to inspect

    Returns:
        EarthLocation: The telescope location
    """
    ms = MS.cast(ms)
    # Get the position of the observatory
    with table(str(ms.mpath / "ANTENNA"), ack=False) as tab:
        pos = EarthLocation.from_geocentric(
            *tab.getcol("POSITION")[0] * u.m  # First antenna is fine
        )
    return pos


def describe_ms(ms: Union[MS, Path], verbose: bool = True) -> MSSummary:
    """Print some basic information from the inpute measurement set.

    Args:
        ms (Union[MS,Path]): Measurement set to inspect
        verbose (bool, optional): Log MS options to the flint logger

    Returns:
        MSSummary: Brief overview of the MS.

    """
    ms = MS(path=ms) if isinstance(ms, Path) else ms

    with table(str(ms.path), readonly=True, ack=False) as tab:
        colnames = tab.colnames()

        flags = tab.getcol("FLAG")
        flagged = np.sum(flags is True)
        unflagged = np.sum(flags is False)
        total = np.prod(flags.shape)

        uniq_ants = sorted(list(set(tab.getcol("ANTENNA1"))))

    with table(f"{ms.path}/FIELD", readonly=True, ack=False) as tab:
        uniq_fields = list(set(tab.getcol("NAME")))

    beam_no = get_beam_from_ms(ms=ms)

    if verbose:
        logger.info(f"Inspecting {ms.path}.")
        logger.info(f"Contains: {colnames}")

        logger.info(f"{flagged} of {total} flagged ({flagged/total*100.:.4f}%). ")
        logger.info(f"{len(uniq_ants)} unique antenna: {uniq_ants}")
        logger.info(f"Unique fields: {uniq_fields}")

    return MSSummary(
        flagged=flagged,
        unflagged=unflagged,
        fields=uniq_fields,
        ants=uniq_ants,
        beam=beam_no,
    )


def split_by_field(
    ms: Union[MS, Path], field: Optional[str] = None, out_dir: Optional[Path] = None
) -> List[MS]:
    """Attempt to split an input measurement set up by the unique FIELDs recorded

    Args:
        ms (Union[MS, Path]): Input measurement sett to split into smaller MSs by field name
        field (Optional[str], optional): Desired field to extract. If None, all are split. Defaults to None.
        out_dir (Optional[Path], optional): Output directory to write the fresh MSs to. If None, write to same directory as
        parent MS. Defaults to None.

    Returns:
        List[MS]: The output MSs split by their field name.
    """
    ms = MS.cast(ms)

    # TODO: Split describe_ms up so can get just fiels
    ms_summary = describe_ms(ms, verbose=False)

    logger.info("Collecting field names and corresponding FIELD_IDs")
    fields = [field] if field else ms_summary.fields
    field_idxs = [get_field_id_for_field(ms=ms, field_name=field) for field in fields]

    out_mss: List[MS] = []

    ms_out_dir: Path = Path(out_dir) if out_dir is not None else ms.path.parent
    logger.info(f"Will write output MSs to {ms_out_dir}.")

    if not ms_out_dir.exists():
        try:
            logger.info(f"Creating {ms_out_dir}.")
            ms_out_dir.mkdir(parents=True)
        except Exception as e:
            logger.warn(e)
            pass  # Incase above fails due to race condition

    logger.info(f"Opening {ms.path}. ")
    with table(str(ms.path), ack=False) as tab:  # noqa: F841
        for split_name, split_idx in zip(fields, field_idxs):
            logger.info(f"Selecting FIELD={split_name}")
            sub_ms = taql(f"select * from $tab where FIELD_ID=={split_idx}")

            out_ms_str = create_ms_name(ms_path=ms.path, field=split_name)
            out_path = ms_out_dir / Path(out_ms_str).name

            logger.info(f"Writing {str(out_path)} for {split_name}")
            sub_ms.copy(str(out_path), deep=True)

            out_mss.append(MS(path=out_path, beam=get_beam_from_ms(out_path)))

    return out_mss


def check_column_in_ms(
    ms: Union[MS, str, PathLike],
    column: Optional[str] = None,
    sub_table: Optional[str] = None,
) -> bool:
    """Checks to see whether a column exists in an MS. If `column` is provided this
    is checked. It `column` is None, then the MS.column is specified. If both are
    `None` then an error is raised.

    Args:
        ms (Union[MS, str, PathLike]): The measurement set to check. Will attempt to cast to Path.
        column (Optional[str], optional): The column to check for. Defaults to None. sub_table (Optional[str], optional): A sub-table of the measurement set to inspect. If `None` the main table is examined. Defaults to None.

    Raises:
        ValueError: Raised when both `column` and `ms.column` are None.

    Returns:
        bool: Whether the column exists in the measurement set.
    """

    check_col = column
    if isinstance(ms, MS):
        logger.debug(f"{ms.column=} {column=}")
        check_col = column if column is not None else ms.column

    if check_col is None:
        raise ValueError(f"No column to check specified: {ms} {column=}.")

    ms_path = ms.path if isinstance(ms, MS) else Path(ms)
    check_table = str(ms_path) if sub_table is None else f"{str(ms_path)}/{sub_table}"

    logger.debug(f"Checking for {check_col} in {check_table}")
    with table(check_table, readonly=True) as tab:
        tab_cols = tab.colnames()
        logger.debug(f"{ms_path} contains columns={tab_cols}.")
        result = check_col in tab_cols

    return result


def consistent_ms(ms1: MS, ms2: MS) -> bool:
    """Perform some basic consistency checks to ensure MS1 can
    be combined with MS2. This is important when considering
    candidate AO-style calibration solution files.

    Args:
        ms1 (MS): The first MS to consider
        ms2 (MS): The second MS to consider

    Returns:
        bool: Whether MS1 is consistent with MS2
    """

    logger.info(f"Comparing ms1={str(ms1.path)} to ms2={(ms2.path)}")
    beam1 = get_beam_from_ms(ms=ms1)
    beam2 = get_beam_from_ms(ms=ms2)

    result = True
    reasons = []
    if beam1 != beam2:
        logger.debug(f"Beams are different: {beam1=} {beam2=}")
        reasons.append(f"{beam1=} != {beam2=}")
        result = False

    freqs1 = get_freqs_from_ms(ms=ms1)
    freqs2 = get_freqs_from_ms(ms=ms2)

    if len(freqs1) != len(freqs2):
        logger.debug(f"Length of frequencies differ: {len(freqs1)=} {len(freqs2)=}")
        reasons.append(f"{len(freqs1)=} != {len(freqs2)=}")
        result = False

    min_freqs1, min_freqs2 = np.min(freqs1), np.min(freqs2)
    if min_freqs1 != min_freqs2:
        logger.debug(f"Minimum frequency differ: {min_freqs1=} {min_freqs2=}")
        reasons.append(f"{min_freqs1=} != {min_freqs2=}")
        result = False

    max_freqs1, max_freqs2 = np.max(freqs1), np.max(freqs2)
    if min_freqs1 != min_freqs2:
        logger.debug(f"Maximum frequency differ: {max_freqs1=} {max_freqs2=}")
        reasons.append(f"{max_freqs1=} != {max_freqs2=}")
        result = False

    if not result:
        logger.info(f"{str(ms1.path)} not compatibale with {str(ms2.path)}, {reasons=}")

    return result


def preprocess_askap_ms(
    ms: Union[MS, Path],
    data_column: str = "DATA",
    instrument_column: str = "INSTRUMENT_DATA",
    overwrite: bool = True,
    skip_rotation: bool = False,
) -> MS:
    """The ASKAP MS stores its data in a way that is not immediatedly accessible
    to other astronomical software, like wsclean or casa. For each measurement set
    the centre of the field is stored, and beam offsets are stored in a separate table.

    Additionally, the correlations stored are more akin to (P, Q) -- they are not
    (X, Y) in the sky reference frame. This function does two things:

    * updates the positions stored so when data are imaged/calibrated the correlations are directed to the correct position
    * will apply a rotation to go from (P, Q) -> (X, Y)

    These corrections are applied to the original MS, and should be
    able to be executed multiple times without accumulating changes.

    Args:
        ms (Union[MS, Path]): The measurement set to update
        data_column (str, optional): The name of the data column to correct. This will first be renamed to the value specified by `instrument_column` before being corrected. Defaults to 'DATA'.
        instrument_column (str, optional): The name of the column that will hold the original `data_column` data. Defaults to 'INSTRUMENT_DATA'
        overwrite (bool, optional): If the `instrument_column` and `data_column` both exist and `overwrite=True` the `data_column` will be overwritten. Otherwise, a `ValueError` is raised. Defaults to True.
        skip_rotation (bool, optional): If true, the visibilities are not rotated Defaults to False.

    Returns:
        MS: An updated measurement set with the corrections applied.
    """
    ms = MS.cast(ms)

    assert (
        data_column != instrument_column
    ), f"Received matching column names: {data_column=} {instrument_column=}"

    logger.info(
        f"Will be running ASKAP MS conversion operations against {str(ms.path)}."
    )
    logger.info("Correcting directions. ")

    with table(str(ms.path), ack=False, readonly=False) as tab:
        colnames = tab.colnames()
        if data_column not in colnames:
            raise ValueError(
                f"Column {data_column} not found in {str(ms.path)}. Columns found: {colnames}"
            )
        if all([col in colnames for col in (data_column, instrument_column)]):
            msg = f"Column {instrument_column} already in {str(ms.path)}. Already corrected?"
            if not overwrite:
                raise ValueError(msg)

        if (
            not skip_rotation
            and data_column in colnames
            and instrument_column not in colnames
        ):
            logger.info(f"Renaming {data_column} to {instrument_column}.")
            tab.renamecol(data_column, instrument_column)

    logger.info("Correcting the field table. ")
    fix_ms_dir(ms=str(ms.path))

    if skip_rotation:
        # TODO: Should we copy the DATA to INSTRUMENT_DATA?
        logger.info("Skipping the rotation of the visibilities. ")
        logger.info(f"Returning {ms=}.")
        return ms.with_options(column=data_column)

    logger.info("Applying roation matrix to correlations. ")
    logger.info(
        f"Rotating visibilities for {ms.path} with data_column={instrument_column} amd corrected_data_column={data_column}"
    )
    fix_ms_corrs(
        ms=ms.path, data_column=instrument_column, corrected_data_column=data_column
    )

    return ms.with_options(column=data_column)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Components to interact with MS")

    subparser = parser.add_subparsers(dest="mode")

    split_parser = subparser.add_parser(
        "split", help="Split an MS based on field name. "
    )

    split_parser.add_argument("ms", type=Path, help="MS to split based on fields. ")
    split_parser.add_argument(
        "--ms-out-dir",
        type=Path,
        default=None,
        help="Location to write the output MSs to. ",
    )

    preprocess_parser = subparser.add_parser(
        "preprocess",
        help="Apply preprocessing operations to the ASKAP MS so it can be used outside of yandasoft",
    )

    preprocess_parser.add_argument("ms", type=Path, help="Measurement set to correct. ")

    compatible_parser = subparser.add_parser(
        "compatible", help="Some basic checks to ensure ms1 is consistent with ms2."
    )
    compatible_parser.add_argument(
        "ms1", type=Path, help="The first measurement set to consider. "
    )
    compatible_parser.add_argument(
        "ms2", type=Path, help="The second measurement set to consider. "
    )

    return parser


def cli() -> None:
    import logging

    logger.setLevel(logging.INFO)

    parser = get_parser()

    args = parser.parse_args()

    if args.mode == "split":
        split_by_field(ms=args.ms, out_dir=args.ms_out_dir)
    if args.mode == "preprocess":
        preprocess_askap_ms(ms=args.ms)
    if args.mode == "compatible":
        res = consistent_ms(ms1=MS(path=args.ms1), ms2=MS(path=args.ms2))
        if res:
            logger.info(f"{args.ms1} is compatible with {args.ms2}")
        else:
            logger.info(f"{args.ms1} is not compatible with {args.ms2}")


if __name__ == "__main__":
    cli()
