"""Figure colors and axis conventions.

Bulk is a solid line. The 3–12 Å interface is dashed. Cooling-rate colors
are the directory-code colors, looked up by the physical rate. Temperature
axes run from hot to cold.
"""

from opcfilm.catalog import RATE_COLOR_BY_VALUE
from opcfilm.config import CONFIG

_interior, _awi, _outer = CONFIG.regions.names
REGION_STYLE = {
    _interior: ("Bulk", "-"),
    _awi: ("AWI", "--"),
    _outer: ("AWI outer", ":"),
}

RATE_COLOR = dict(RATE_COLOR_BY_VALUE)


def temperature_axis(axis):
    """Put high temperature on the left.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axis whose x data are temperatures.

    Returns
    -------
    None
        The x axis is inverted.
    """
    axis.invert_xaxis()
