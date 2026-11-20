"""One energy table, then a per-window or a binned view.

Records come from the single mdout parser. Averages printed after the
AVERAGES banner are not records.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from opcfilm.config import CONFIG
from opcfilm.io.mdout import parse_dataframe


def records(path):
    """Per-record energy table of one mdout.

    Parameters
    ----------
    path : path-like
        Amber mdout.

    Returns
    -------
    pandas.DataFrame
        One row per printed step, with the duplicate step dropped.
    """
    return parse_dataframe(path)


def per_window(frame, temperature_column="temp_K"):
    """Mean energy of each temperature already present in the records.

    Parameters
    ----------
    frame : pandas.DataFrame
        Output of :func:`records`.
    temperature_column : str, optional
        Temperature column.

    Returns
    -------
    pandas.DataFrame
        One row per rounded kelvin, with the mean ``EPtot``.
    """
    grouped = frame.groupby(frame[temperature_column].round().astype(int))
    return grouped["EPtot"].mean().rename("EPtot").reset_index()


def binned(frame, width_K=None, temperature_column="temp_K"):
    """Mean energy in bins of ``width_K``.

    Parameters
    ----------
    frame : pandas.DataFrame
        Output of :func:`records`.
    width_K : float, optional
        Bin width in kelvin. The sheet value is 2 K.
    temperature_column : str, optional
        Temperature column.

    Returns
    -------
    pandas.DataFrame
        Columns ``Tbin`` and ``EPtot``.
    """
    if width_K is None:
        width_K = CONFIG.energy.bin_K
    temperature = frame[temperature_column].to_numpy(dtype=float)
    edges = np.floor(temperature / float(width_K)) * float(width_K)
    copied = frame.copy()
    copied["Tbin"] = edges + float(width_K) / 2.0
    return copied.groupby("Tbin", as_index=False)["EPtot"].mean()
