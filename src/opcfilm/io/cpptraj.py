"""Read cpptraj standard data files and the surftension outputs.

cpptraj writes one header line, ``#<x label> <legend> <legend> ...``, and one
row per X value. The X label can contain spaces (``q (Ang^-1)``), so columns
are named from the legends: every token of the form ``NAME[aspect]`` is a
data set, and the column takes the aspect (``wtop``, ``gammaq``). Sets with
different X dimensions are written as blocks separated by a blank line; each
block is read on its own and the blocks are merged on X.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

_LEGEND = re.compile(r"(\S+)\[([^\]]+)\]")
_HELFRICH = re.compile(r"gamma,\s*Helfrich intercept\s*=\s*(\S+)\s*mN/m")


def _block_frame(header, rows):
    legends = _LEGEND.findall(header or "")
    width = max(len(r) for r in rows)
    rows = [r + [np.nan] * (width - len(r)) for r in rows]
    data = np.asarray(rows, dtype=float)
    if legends and len(legends) == width - 1:
        names = ["x"] + [aspect for _name, aspect in legends]
    elif legends and len(legends) == width:
        names = [aspect for _name, aspect in legends]
    else:
        names = ["x"] + [f"c{i}" for i in range(1, width)]
    return pd.DataFrame(data, columns=names)


def read_sets(path):
    """Data sets of one cpptraj file, one column per set aspect.

    Parameters
    ----------
    path : path-like
        A file written by cpptraj ``out``/``spectrumout``/``roughout``/``blockout``.

    Returns
    -------
    pandas.DataFrame
        Column ``x`` (frame, block, or q) and one column per set, named by the
        aspect in brackets of its legend.
    """
    blocks, header, rows = [], None, []
    for line in Path(path).read_text().splitlines():
        text = line.strip()
        if not text:
            if rows:
                blocks.append(_block_frame(header, rows))
                header, rows = None, []
            continue
        if text.startswith("#"):
            if rows:
                blocks.append(_block_frame(header, rows))
                rows = []
            header = text.lstrip("#")
            continue
        try:
            rows.append([float(value) for value in text.split()])
        except ValueError:
            continue
    if rows:
        blocks.append(_block_frame(header, rows))
    if not blocks:
        raise ValueError(f"no numeric rows in {path}")
    merged = blocks[0]
    for block in blocks[1:]:
        merged = merged.merge(block, on="x", how="outer") if "x" in block else pd.concat([merged, block], axis=1)
    return merged


def read_summary(path):
    """Key/value pairs of a surftension ``summaryout`` file."""
    values = {}
    for line in Path(path).read_text().splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        parts = text.split(None, 1)
        if len(parts) != 2:
            continue
        try:
            values[parts[0]] = float(parts[1])
        except ValueError:
            values[parts[0]] = parts[1]
    return values


def read_helfrich_intercept(log_path):
    """gamma_0 (mN/m) printed in the cpptraj log, or NaN."""
    path = Path(log_path)
    if not path.is_file():
        return math.nan
    found = _HELFRICH.findall(path.read_text(errors="replace"))
    try:
        return float(found[-1]) if found else math.nan
    except ValueError:
        return math.nan


def helfrich_fit(q, gamma_q, qmin, qmax):
    """Fit gamma(q) = gamma_0 + kappa' q^2 over ``qmin <= q <= qmax``.

    Returns
    -------
    tuple of float
        ``(gamma_0, kappa_prime, n_shells)``; kappa' in mN m^-1 Å^2. NaN when
        fewer than three shells are finite in the range.
    """
    q = np.asarray(q, float)
    g = np.asarray(gamma_q, float)
    keep = np.isfinite(q) & np.isfinite(g) & (q >= qmin - 1e-12) & (q <= qmax + 1e-12) & (q > 0)
    if keep.sum() < 3:
        return math.nan, math.nan, int(keep.sum())
    slope, intercept = np.polyfit(q[keep] ** 2, g[keep], 1)
    return float(intercept), float(slope), int(keep.sum())
