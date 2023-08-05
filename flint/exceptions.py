class MSError(Exception):
    """An error for MS related things"""

    pass

class PhaseOutlierFitError(Exception):
    """Raised when the phase outlier fit routine fails. """
    pass