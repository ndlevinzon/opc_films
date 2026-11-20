"""Geometric free-volume probe.

A probe center is free when it is at least ``probe radius + oxygen exclusion``
from every oxygen. The reported radius is 0.80 Å. This is an insertion
fraction, not a pore volume.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from opcfilm.config import CONFIG


def exclusion_distance(probe_radius, oxygen_exclusion=None):
    """Minimum oxygen–probe distance counted as free.

    Parameters
    ----------
    probe_radius : float
        Probe radius in angstroms.
    oxygen_exclusion : float, optional
        Oxygen exclusion radius. The sheet value is the default.

    Returns
    -------
    float
        Sum of the two radii, in angstroms.
    """
    if oxygen_exclusion is None:
        oxygen_exclusion = CONFIG.probe.oxygen_exclusion_A
    return float(probe_radius) + float(oxygen_exclusion)


def probe_fraction(oxygen, lengths, center, intervals, n_probes, exclusion, rng):
    """Fraction of uniform probe centers outside the oxygen exclusion sphere.

    Parameters
    ----------
    oxygen : numpy.ndarray, shape (n, 3)
        Oxygen coordinates in angstroms.
    lengths : array_like, shape (3,)
        Box lengths of this frame.
    center : float
        Slab center in angstroms.
    intervals : sequence of tuple
        ``z`` intervals relative to the slab center.
    n_probes : int
        Number of probe centers.
    exclusion : float
        Minimum oxygen–probe distance counted as free, in angstroms.
    rng : numpy.random.Generator
        Sampler for the probe centers.

    Returns
    -------
    float
        Free fraction. NaN when an interval has no thickness.
    """
    lengths = np.asarray(lengths, dtype=float)
    widths = np.array([high - low for low, high in intervals], dtype=float)
    if np.any(widths <= 0):
        return np.nan
    tree = cKDTree(np.mod(oxygen, lengths), boxsize=lengths)
    choice = rng.choice(len(intervals), size=n_probes, p=widths / widths.sum())
    low = np.array([intervals[index][0] for index in choice])
    high = np.array([intervals[index][1] for index in choice])
    points = np.column_stack([
        rng.uniform(0.0, lengths[0], n_probes),
        rng.uniform(0.0, lengths[1], n_probes),
        np.mod(center + rng.uniform(low, high), lengths[2]),
    ])
    return float(np.mean(tree.query(points, k=1)[0] >= exclusion))
