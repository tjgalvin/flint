"""Tests around utility helper functions for self-calibration"""

from flint.selfcal.utils import consider_skip_selfcal_on_round


def test_consider_skip_selfcal():
    """Ensure that the skipping behaves as expected"""

    # None here means nothing should be skipped
    res = consider_skip_selfcal_on_round(current_round=1, skip_selfcal_on_rounds=None)

    assert not res

    res = consider_skip_selfcal_on_round(current_round=1, skip_selfcal_on_rounds=[2, 3])
    assert not res

    res = consider_skip_selfcal_on_round(current_round=2, skip_selfcal_on_rounds=[2, 3])
    assert res
    res = consider_skip_selfcal_on_round(current_round=2, skip_selfcal_on_rounds=2)
    assert res
