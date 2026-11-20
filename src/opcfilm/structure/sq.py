"""In-plane structure-factor peak.

The grid, the sinc correction, and the frame bootstrap stay with the
structure-factor script. The quadratic peak location is the shared step.
"""

from __future__ import annotations

import numpy as np

from opcfilm.config import QSTAR_JSON


def peak_position(q, intensity):
    """Vertex of a quadratic fit to ``intensity(q)``.

    Parameters
    ----------
    q : array_like
        Wavevector in inverse angstroms.
    intensity : array_like
        Structure factor on those points. At least three finite points are
        required, and the curvature must be negative.

    Returns
    -------
    float
        Peak position in inverse angstroms.

    Raises
    ------
    ValueError
        If the fit has no maximum inside the supplied points.
    """
    q = np.asarray(q, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    finite = np.isfinite(q) & np.isfinite(intensity)
    q = q[finite]
    intensity = intensity[finite]
    if len(q) < 3:
        raise ValueError("A quadratic peak fit needs at least three points.")
    curvature, slope, _intercept = np.polyfit(q, intensity, 2)
    if curvature >= 0:
        raise ValueError("The quadratic fit has no maximum.")
    position = float(-slope / (2.0 * curvature))
    if position < float(np.min(q)) or position > float(np.max(q)):
        raise ValueError("The quadratic maximum is outside the fit window.")
    return position


def qstar_path(root):
    """Path of the one adopted-q* JSON.

    Parameters
    ----------
    root : path-like
        Repository or run directory.

    Returns
    -------
    pathlib.Path
        ``<root>/<QSTAR_JSON>``.
    """
    from pathlib import Path

    return Path(root) / QSTAR_JSON
