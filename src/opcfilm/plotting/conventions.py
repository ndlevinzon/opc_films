"""House conventions applied to every figure when it is saved.

Two rules hold for all figures in the project:

1. A temperature axis reads from hot on the left to cold on the right.
2. A legend never sits inside a plotting window.

:func:`install_figure_conventions` wraps :meth:`matplotlib.figure.Figure.savefig`
so that both rules are applied just before any figure is written, whichever
script drew it. Plotting scripts call it once, after importing pyplot.

An x axis counts as a temperature axis when its label names a temperature
in kelvin, for example ``Temperature (K)`` or ``Cooling-window midpoint (K)``.
Inverse temperature (``1000/T``), cooling rate, and temperature differences
are not temperature axes: ``1000/T`` already runs hot to cold. Overlay axes
made with ``twinx`` or ``twiny`` keep the scale of the axes they overlay,
because their ticks are placed in that axes' coordinates.

When no panel with a legend has another panel to its right, each legend
moves just outside its panel's right edge. Otherwise all panel legends are
merged into one figure legend to the right of all panels, with repeated
labels shown once. Figure legends move to the right of the figure.
Figures are saved with ``bbox_inches="tight"`` unless the caller says
otherwise, so legends outside the panels are not cut off.
"""

from __future__ import annotations

import re

_TEMPERATURE = re.compile(r"temperature|midpoint|\(\s*K\s*\)\s*\$?\s*$", re.IGNORECASE)
_NOT_TEMPERATURE = re.compile(
    r"1000\s*/|1\s*/\s*T|\^\{?-1|K\s*/\s*ns|per\s+K|\\Delta|Δ|difference",
    re.IGNORECASE,
)
_OUTSIDE_GAP = 0.02


def is_temperature_label(label):
    """Whether an axis label names a temperature in kelvin.

    Parameters
    ----------
    label : str
        Axis label text.

    Returns
    -------
    bool
        True for labels such as ``Temperature (K)``; False for ``1000/T``,
        cooling rates, temperature differences, and empty labels.
    """
    if not label:
        return False
    return bool(_TEMPERATURE.search(label)) and not _NOT_TEMPERATURE.search(label)


def _panels(figure):
    """Plot axes grouped by position, colorbars left out.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure to inspect.

    Returns
    -------
    list of list of matplotlib.axes.Axes
        One list per panel. The first axes of a list is the primary one and
        the rest overlay it.
    """
    panels = []
    for axis in figure.axes:
        if axis.get_label() == "<colorbar>" or not axis.get_visible():
            continue
        box = axis.get_position().bounds
        for panel in panels:
            other = panel[0].get_position().bounds
            if all(abs(a - b) < 1e-6 for a, b in zip(box, other)):
                panel.append(axis)
                break
        else:
            panels.append([axis])
    return panels


def _has_right_neighbour(panel, panels):
    """Whether another panel lies to the right and overlaps vertically.

    Parameters
    ----------
    panel : list of matplotlib.axes.Axes
        Panel to test.
    panels : list of list of matplotlib.axes.Axes
        All panels of the figure.

    Returns
    -------
    bool
        True when a legend outside the right edge would cover a panel.
    """
    box = panel[0].get_position()
    for other in panels:
        if other is panel:
            continue
        candidate = other[0].get_position()
        overlaps = candidate.y0 < box.y1 and candidate.y1 > box.y0
        if overlaps and candidate.x0 >= box.x1 - 1e-6:
            return True
    return False


def _set_loc(legend, location):
    """Set a legend's location on any recent Matplotlib.

    Parameters
    ----------
    legend : matplotlib.legend.Legend
        Legend to move.
    location : str
        Matplotlib location string.

    Returns
    -------
    None
        The legend is changed in place.
    """
    if hasattr(legend, "set_loc"):
        legend.set_loc(location)
    else:
        legend._set_loc(legend.codes[location])


def _legend_entries(legend):
    """Handles and labels shown by a legend.

    Parameters
    ----------
    legend : matplotlib.legend.Legend
        Existing legend.

    Returns
    -------
    list of tuple
        ``(handle, label)`` in display order.
    """
    handles = getattr(legend, "legend_handles", None)
    if handles is None:
        handles = legend.legendHandles
    labels = [text.get_text() for text in legend.get_texts()]
    return list(zip(handles, labels))


def apply_figure_conventions(figure):
    """Put temperature axes hot to cold and legends outside the panels.

    Calling it again on the same figure does nothing, so a figure saved as
    both PNG and PDF is changed once.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure about to be saved.

    Returns
    -------
    None
        The figure is changed in place.
    """
    if getattr(figure, "_opc_conventions_applied", False):
        return
    panels = _panels(figure)

    for panel in panels:
        primary = panel[0]
        if is_temperature_label(primary.get_xlabel()) and not primary.xaxis_inverted():
            primary.invert_xaxis()

    with_legends = []
    for panel in panels:
        legends = [axis.get_legend() for axis in panel if axis.get_legend() is not None]
        if legends:
            with_legends.append((panel, legends))
    # One crowded panel means every panel legend joins the figure legend, so
    # no two legends compete for the space right of the panels.
    merge = any(_has_right_neighbour(panel, panels) for panel, _ in with_legends)
    merged = []
    for panel, legends in with_legends:
        if merge:
            for legend in legends:
                merged.extend(_legend_entries(legend))
                legend.remove()
            continue
        y = 1.0
        for legend in legends:
            _set_loc(legend, "upper left")
            legend.set_bbox_to_anchor((1.0 + _OUTSIDE_GAP, y), transform=panel[0].transAxes)
            y -= 0.5

    for legend in list(figure.legends):
        _set_loc(legend, "center left")
        legend.set_bbox_to_anchor((1.0, 0.5), transform=figure.transFigure)

    if merged:
        seen, handles, labels = set(), [], []
        for handle, label in merged:
            if label in seen or label.startswith("_"):
                continue
            seen.add(label)
            handles.append(handle)
            labels.append(label)
        right = max(panel[0].get_position().x1 for panel in panels)
        figure.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(right + _OUTSIDE_GAP, 0.5),
            bbox_transform=figure.transFigure,
            frameon=False,
        )
    figure._opc_conventions_applied = True


def install_figure_conventions():
    """Apply :func:`apply_figure_conventions` whenever a figure is saved.

    Safe to call more than once. ``pyplot.savefig`` saves through
    :meth:`Figure.savefig`, so it is covered too.

    Returns
    -------
    None
        ``matplotlib.figure.Figure.savefig`` is wrapped.
    """
    from matplotlib.figure import Figure

    if getattr(Figure.savefig, "_opc_conventions", False):
        return
    original = Figure.savefig

    def savefig(self, *args, **kwargs):
        apply_figure_conventions(self)
        kwargs.setdefault("bbox_inches", "tight")
        return original(self, *args, **kwargs)

    savefig._opc_conventions = True
    savefig.__doc__ = original.__doc__
    Figure.savefig = savefig
