"""Basic tests around prefect helper functions"""

from __future__ import annotations

from prefect import flow

from flint.prefect.helpers import enable_loguru_support


def test_enable_loguru_support():
    """Some packages may be using loguru (e.g. crystalball). Should
    we want those logs to be captured we need to modify the loguru
    logger. A helpful function has been added to this end. This
    is a simple, very basic test to make sure it can still run without
    error, though whether it still works is a completely different
    question!"""

    @flow
    def example_flow():
        enable_loguru_support()

    example_flow()
