"""Operations related to measurement sets
"""
from __future__ import (
    annotations,
)  # Used for mypy/pylance to like the return type of MS.with_options

from os import PathLike
from pathlib import Path
from typing import NamedTuple, Optional, Union

from casacore.tables import table

from flint.logging import logger


class MS(NamedTuple):
    """Helper to keep tracked of measurement set information"""

    path: Path
    column: Optional[str] = None
    beam: Optional[int] = None

    def with_options(self, **kwargs) -> MS:
        as_dict = self._asdict()
        as_dict.update(kwargs)

        return MS(**as_dict)


# TODO: Some common MS validation functions?
# - list / number of fields
# - new name function (using names / beams)
# - check to see if fix_ms_dir / fix_ms_corrs
# - delete column/rename column


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
