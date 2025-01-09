"""Gemeral help functions that could be used for self-calibration
across different packages.
"""

from __future__ import annotations


def consider_skip_selfcal_on_round(
    current_round: int, skip_selfcal_on_rounds: int | list[int] | None
) -> bool:
    """Consider whether the self-calibration process (derive and applying solutions)
    should be skipped on a particular imaging round.

    Should `current_round` be in `skip_selfcal_on_round` then the self-calibration
    should be skipped and a `True` is returned.

    Args:
        current_round (int): The current imaging round being considered
        skip_selfcal_on_rounds (Union[int, List[int], None]): The set of rounds that should be considerede for skipping. If None a False is returned.

    Returns:
        bool: Whether the round is skipped (True) or performed (False)
    """

    # Consider whether this is unset, in which case all rounds should be self-cal
    if skip_selfcal_on_rounds is None:
        return False

    # For sanity consider a single int case
    skip_selfcal_on_rounds = (
        skip_selfcal_on_rounds
        if isinstance(skip_selfcal_on_rounds, list)
        else [skip_selfcal_on_rounds]
    )

    return current_round in skip_selfcal_on_rounds
