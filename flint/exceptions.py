class MSError(Exception):
    """An error for MS related things"""

    pass


class PhaseOutlierFitError(Exception):
    """Raised when the phase outlier fit routine fails."""

    pass


class GainCalError(Exception):
    """Raised when it appears like the casa gaincal task fails."""

    pass


class CleanDivergenceError(Exception):
    """Raised if it is detected that cleaning has diverged."""

    pass


class TarArchiveError(Exception):
    """Raised it the flint tarball is not created successfullty"""
