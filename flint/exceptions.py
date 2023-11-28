class MSError(Exception):
    """An error for MS related things"""

    pass


class PhaseOutlierFitError(Exception):
    """Raised when the phase outlier fit routine fails."""

    pass


class GainCalError(Exception):
    """Raised when it appears like the casa gaincal task fails."""

    pass
