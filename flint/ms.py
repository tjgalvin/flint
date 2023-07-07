"""Operations related to measurement sets
"""
from pathlib import Path
from typing import NamedTuple, Optional

from casacore.tables import table

from flint.logging import logger


class MS(NamedTuple):
    """Helper to keep tracked of measurement set information"""

    path: Path
    column: Optional[str] = None


# TODO: Some common MS validation functions?


def check_column_in_ms(ms: MS, column: Optional[str] = None) -> bool:
    """Checks to see whether a column exists in an MS. If `column` is provided this
    is checked. It `column` is None, then the MS.column is specified. If both are
    `None` then an error is raised.

    Args:
        ms (MS): The measurement set to check
        column (Optional[str], optional): The column to check for. Defaults to None.

    Raises:
        ValueError: Raised when both `column` and `ms.column` are None.

    Returns:
        bool: Whether the column exists in the measurement set.
    """
    # TODO: Support sub-tables

    check_col = column if column is not None else ms.column
    if check_col is None:
        raise ValueError(
            f"No column to check specified: {ms.path=} {ms.column=} {column=}"
        )

    logger.debug(f"Checking for {check_col}")
    with table(str(ms.path), readonly=True) as tab:
        tab_cols = tab.colnames()
        logger.debug(f"{ms.path} contains columns={tab_cols}.")
        if check_col in tab_cols:
            return True

        return False
