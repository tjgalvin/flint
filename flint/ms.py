"""Operations related to measurement sets
"""
from __future__ import (
    annotations,
)  # Used for mypy/pylance to like the return type of MS.with_options

from os import PathLike
from pathlib import Path
from typing import NamedTuple, Optional, Union, List

import numpy as np
from casacore.tables import table

from flint.logging import logger


class MS(NamedTuple):
    """Helper to keep tracked of measurement set information"""

    path: Path
    column: Optional[str] = None
    beam: Optional[int] = None
    field: Optional[str] = None

    def get_field_id_for_field(self, field_name: str) -> Union[int, None]:
        """Return the FIELD_ID for an elected field in a measurement set. See
        `flink.ms.get_field_id_for_field` for full details.
        """

        return get_field_id_for_field(ms=self, field_name=field_name)

    def with_options(self, **kwargs) -> MS:
        """Create a new MS instance with keywords updated

        Returns:
            MS: New MS instance with updated attributes
        """
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

    with table(str(ms_path), readonly=True) as tab:
        uniq_beams = sorted(list(set(tab.getcol("FEED1"))))

    assert (
        len(uniq_beams) == 1
    ), f"Expected {str(ms_path)} to contain a single beam, found {len(uniq_beams)}: {uniq_beams=}"

    return uniq_beams[0]


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
        flagged = np.sum(flags == True)
        unflagged = np.sum(flags == False)
        total = np.prod(flags.shape)

        uniq_ants = sorted(list(set(tab.getcol("ANTENNA1"))))

    with table(f"{ms.path}/FIELD", readonly=True) as tab:
        uniq_fields = sorted(list(set(tab.getcol("NAME"))))

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
        column (Optional[str], optional): The column to check for. Defaults to None.
        sub_table (Optional[str], optional): A sub-table of the measurement set to inspect. If `None`
        the main table is examined. Defaults to None.

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
