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

Region styles follow the series label. Bulk (also labelled interior or
core) is solid, AWI is dashed, and the outer AWI band (0–6 Å) is dotted.
Series drawn with markers only show bulk as filled and AWI as open
markers. A label that names both, such as ``Bulk − AWI``, is a contrast and
keeps its style. Legend text says ``Bulk`` and ``AWI`` rather than
``interior`` and ``awi``.

A panel that compares at most two labelled series is drawn in black, with
grey uncertainty bands. If the two series would then look the same, the
second is dashed, or drawn with open markers. Panels with an overlay axis
(``twinx`` or ``twiny``) keep their colors, which identify the axes.
"""

from __future__ import annotations

import re

_TEMPERATURE = re.compile(r"temperature|midpoint|\(\s*K\s*\)\s*\$?\s*$", re.IGNORECASE)
_NOT_TEMPERATURE = re.compile(
    r"1000\s*/|1\s*/\s*T|\^\{?-1|K\s*/\s*ns|per\s+K|\\Delta|Δ|difference",
    re.IGNORECASE,
)
_OUTSIDE_GAP = 0.02

_BULK = re.compile(r"\b(bulk|interior|core)\b", re.IGNORECASE)
_AWI = re.compile(r"\bawi\b|interface", re.IGNORECASE)
_OUTER = re.compile(r"outer|\b0\s*[–-]\s*6\b", re.IGNORECASE)
_REGION_LINESTYLE = {"bulk": "-", "awi": "--", "awi_outer": ":"}
_LABEL_WORDS = ((re.compile(r"\binterior\b", re.IGNORECASE), "Bulk"), (re.compile(r"\bawi\b"), "AWI"))


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


def region_of(label):
    """Region a series label refers to.

    Parameters
    ----------
    label : str
        Series label.

    Returns
    -------
    str or None
        ``"bulk"``, ``"awi"`` or ``"awi_outer"``; None for unlabelled
        series, other labels, and contrasts that name both regions.
    """
    if not label or label.startswith("_"):
        return None
    bulk, awi = _BULK.search(label), _AWI.search(label)
    if bulk and awi:
        return None
    if awi:
        return "awi_outer" if _OUTER.search(label) else "awi"
    if bulk:
        return "bulk"
    return None


def normalize_label(label):
    """Use Bulk and AWI in place of interior and awi.

    Parameters
    ----------
    label : str
        Series or legend label.

    Returns
    -------
    str
        The label with the region words replaced.
    """
    for pattern, word in _LABEL_WORDS:
        label = pattern.sub(word, label)
    return label


def _series(axis):
    """Labelled data series of one axes.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axes to inspect.

    Returns
    -------
    list of tuple
        ``(kind, artist, label)`` with kind ``"errorbar"``, ``"line"`` or
        ``"scatter"``. Lines that belong to an errorbar are reported once,
        through the errorbar.
    """
    from matplotlib.collections import PathCollection
    from matplotlib.container import ErrorbarContainer

    found, owned = [], set()
    for container in axis.containers:
        if isinstance(container, ErrorbarContainer):
            for child in container.get_children():
                owned.add(id(child))
            label = container.get_label() or ""
            if label and not label.startswith("_"):
                found.append(("errorbar", container, label))
    for line in axis.get_lines():
        label = line.get_label() or ""
        if id(line) in owned or not label or label.startswith("_"):
            continue
        found.append(("line", line, label))
    for collection in axis.collections:
        label = collection.get_label() or ""
        if isinstance(collection, PathCollection) and label and not label.startswith("_"):
            found.append(("scatter", collection, label))
    return found


def _marker_only(line):
    """Whether a line is drawn with markers and no connecting line."""
    return line.get_linestyle() in ("None", "none", "", " ") and line.get_marker() not in ("None", "none", "", " ", None)


def _style_region(kind, artist, region):
    """Apply the line style or marker fill of a region."""
    if kind == "errorbar":
        artist = artist.lines[0]
        kind = "line"
        if artist is None:
            return
    if kind == "line":
        if _marker_only(artist):
            if region.startswith("awi"):
                artist.set_markerfacecolor("none")
        else:
            artist.set_linestyle(_REGION_LINESTYLE[region])
    elif kind == "scatter" and region.startswith("awi"):
        colors = artist.get_facecolors()
        # Already hollow markers keep their edge color.
        if len(colors) and any(color[3] > 0 for color in colors):
            artist.set_edgecolors(colors)
            artist.set_facecolors("none")


def _signature(kind, artist):
    """What tells one black series from another."""
    if kind == "errorbar":
        artist = artist.lines[0]
        kind = "line"
        if artist is None:
            return ("errorbar",)
    if kind == "line":
        return (artist.get_linestyle(), artist.get_marker(), str(artist.get_markerfacecolor()))
    faces = artist.get_facecolors()
    return ("scatter", str(artist.get_paths()[0].vertices.round(3).tolist()) if artist.get_paths() else "", len(faces) == 0)


def _blacken(kind, artist):
    """Draw a series in black."""
    if kind == "errorbar":
        for child in artist.get_children():
            if hasattr(child, "set_color"):
                child.set_color("black")
            # Caps are markers, colored by their edge.
            if hasattr(child, "set_markeredgecolor"):
                child.set_markeredgecolor("black")
        line = artist.lines[0]
        if line is not None and str(line.get_markerfacecolor()) != "none":
            line.set_markerfacecolor("black")
        return
    if kind == "line":
        artist.set_color("black")
        if str(artist.get_markerfacecolor()) != "none":
            artist.set_markerfacecolor("black")
        artist.set_markeredgecolor("black")
        return
    hollow = len(artist.get_facecolors()) == 0 or all(c[3] == 0 for c in artist.get_facecolors())
    artist.set_edgecolors("black")
    artist.set_facecolors("none" if hollow else "black")


def _differentiate(kind, artist):
    """Make the second of two identical black series distinguishable."""
    if kind == "errorbar":
        artist = artist.lines[0]
        kind = "line"
        if artist is None:
            return
    if kind == "line":
        if _marker_only(artist):
            artist.set_markerfacecolor("none")
        else:
            artist.set_linestyle("--")
    else:
        artist.set_facecolors("none")


def _style_series(panels):
    """Region styles, black two-series panels, and Bulk/AWI labels."""
    for panel in panels:
        series = []
        for axis in panel:
            for kind, artist, label in _series(axis):
                artist.set_label(normalize_label(label))
                region = region_of(label)
                if region is not None:
                    _style_region(kind, artist, region)
                series.append((kind, artist))
        if len(panel) > 1 or not 0 < len(series) <= 2:
            continue
        for kind, artist in series:
            _blacken(kind, artist)
        # Error bars drawn without a label belong to the black series.
        from matplotlib.container import ErrorbarContainer
        for container in panel[0].containers:
            if isinstance(container, ErrorbarContainer):
                _blacken("errorbar", container)
        if len(series) == 2 and _signature(*series[0]) == _signature(*series[1]):
            _differentiate(*series[1])
        for collection in panel[0].collections:
            # Uncertainty bands from fill_between.
            if type(collection).__name__ in ("PolyCollection", "FillBetweenPolyCollection"):
                collection.set_facecolor("0.6")
                collection.set_edgecolor("none")


def _sync_legend(legend, panel):
    """Match legend entries to the restyled series they stand for."""
    by_label = {}
    for axis in panel:
        for kind, artist, label in _series(axis):
            by_label[label] = (kind, artist)
    from matplotlib.offsetbox import DrawingArea

    def drawing_areas(box):
        for child in box.get_children():
            if isinstance(child, DrawingArea):
                yield child
            elif hasattr(child, "get_children"):
                yield from drawing_areas(child)

    areas = list(drawing_areas(legend))
    handles = _handles(legend)
    for index, (handle, text) in enumerate(zip(handles, legend.get_texts())):
        label = normalize_label(text.get_text())
        text.set_text(label)
        if label not in by_label or handle is None:
            continue
        kind, artist = by_label[label]
        source = artist.lines[0] if kind == "errorbar" else artist
        if source is None:
            continue
        try:
            if kind == "errorbar":
                # An errorbar entry draws its bar and marker as separate artists.
                parts = list(areas[index].get_children()) if index < len(areas) else [handle]
                for part in parts:
                    if hasattr(part, "set_color"):
                        part.set_color(source.get_color())
                    if hasattr(part, "set_markerfacecolor"):
                        part.set_markerfacecolor(source.get_markerfacecolor())
                        part.set_markeredgecolor(source.get_markeredgecolor())
            elif kind == "line" and hasattr(handle, "set_linestyle"):
                # Copy style only: update_from would also copy the data transform.
                if hasattr(handle, "set_marker"):
                    handle.set_color(source.get_color())
                    handle.set_linestyle(source.get_linestyle())
                    handle.set_marker(source.get_marker())
                    handle.set_markerfacecolor(source.get_markerfacecolor())
                    handle.set_markeredgecolor(source.get_markeredgecolor())
                else:
                    handle.set_color(source.get_color())
            elif kind == "scatter":
                handle.set_facecolors(source.get_facecolors())
                handle.set_edgecolors(source.get_edgecolors())
        except (AttributeError, TypeError, ValueError):
            pass


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


def _colorbars(figure):
    """Colorbar axes of a figure."""
    return [axis for axis in figure.axes if axis.get_label() == "<colorbar>" and axis.get_visible()]


def _has_right_neighbour(panel, panels, others=()):
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
    for other in list(panels) + [[axis] for axis in others]:
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


def _handles(legend):
    """Legend handles under the current and the older Matplotlib name."""
    handles = getattr(legend, "legend_handles", None)
    if handles is None:
        handles = getattr(legend, "legendHandles", [])
    return list(handles)


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
    handles = _handles(legend)
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

    _style_series(panels)
    for panel in panels:
        for axis in panel:
            if axis.get_legend() is not None:
                _sync_legend(axis.get_legend(), panel)

    with_legends = []
    for panel in panels:
        legends = [axis.get_legend() for axis in panel if axis.get_legend() is not None]
        if legends:
            with_legends.append((panel, legends))
    # One crowded panel means every panel legend joins the figure legend, so
    # no two legends compete for the space right of the panels.
    # A colorbar next to a panel also takes the space a legend would use.
    bars = _colorbars(figure)
    merge = any(_has_right_neighbour(panel, panels, bars) for panel, _ in with_legends)
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
        right = max([panel[0].get_position().x1 for panel in panels]
                    + [axis.get_tightbbox(figure.canvas.get_renderer()).transformed(figure.transFigure.inverted()).x1
                       for axis in bars])
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
