"""Manuscript figures from reduced tables.

Each figure module calls :func:`save` with the metric it displays. The
temperature axis runs from hot to cold.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from opcfilm.plotting.conventions import install_figure_conventions
install_figure_conventions()  # temperature axes hot to cold; legends outside panels
import pandas as pd

from opcfilm.plotting.style import temperature_axis


def save(frame, metric, path, title):
    """Draw one metric against temperature.

    Parameters
    ----------
    frame : pandas.DataFrame
        A reduced table with ``temperature_K``, ``metric``, and ``value``.
    metric : str
        Metric to draw. The function refuses a table that does not contain it.
    path : path-like
        PDF destination.
    title : str
        Figure title.

    Returns
    -------
    pathlib.Path
        The PDF.

    Raises
    ------
    ValueError
        If ``metric`` is not a column value. Another metric is not substituted.
    """
    frame = frame.copy()
    if "metric" not in frame.columns or metric not in set(frame["metric"]):
        raise ValueError(f"The table has no metric {metric}.")
    chosen = frame[frame["metric"] == metric]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots()
    temperature = pd.to_numeric(chosen["temperature_K"], errors="coerce")
    value = pd.to_numeric(chosen["value"], errors="coerce")
    axis.plot(temperature, value, marker="o")
    axis.set_xlabel("Temperature (K)")
    axis.set_ylabel(str(chosen["metric"].iloc[0]))
    axis.set_title(title)
    if temperature.nunique() > 1:
        temperature_axis(axis)
    figure.savefig(path)
    plt.close(figure)
    return path


