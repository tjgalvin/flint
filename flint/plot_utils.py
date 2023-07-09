"""Some potentially useful plotting helpers
"""
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from flint.logging import logger


def fill_between_flags(
    ax: plt.Axes,
    flags: np.ndarray,
    values: Optional[np.ndarray] = None,
    direction: str = "x",
) -> None:
    """Plot vertical or horizontal lines where data are flagged.

    NOTE: This is pretty inefficent and not intended for regular use.

    Args:
        ax (plt.Axes): Axes object to plot lines on
        flags (np.ndarray): Flags to consider. If `True`, plot.
        values (Optional[np.ndarray], optional): The values to plot at. Useful if the position does not map to location.. Defaults to None.
        direction (str, optional): If `x` use axvline, if `y` use axhline. Defaults to "x".
    """
    values = values if values else np.arange(len(flags))

    mask = np.argwhere(flags)
    plot_vals = values[mask]
    func = ax.axvline if direction == "x" else ax.axhline

    for v in plot_vals:
        func(v, color="black", alpha=0.3)
