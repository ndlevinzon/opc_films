"""Hydrogen-bond counts, directed pairs, and retention.

The geometry is the sheet cutoff and angle. Retention is the fraction of
origin pairs that are still present at a later lag.
"""

from __future__ import annotations

import numpy as np

from opcfilm.geometry.slab import directed_hydrogen_bonds, hydrogen_bond_counts


def intermittent_retention(origin_pairs, later_pairs):
    """Fraction of origin pairs that are present again at a later frame.

    Parameters
    ----------
    origin_pairs : collection of tuple
        Directed ``(donor, acceptor)`` pairs at the origin.
    later_pairs : collection of tuple
        Directed pairs at the lag.

    Returns
    -------
    float
        Retained fraction. NaN when the origin has no pairs.
    """
    origin = set(origin_pairs)
    if not origin:
        return np.nan
    later = set(later_pairs)
    return float(len(origin & later) / len(origin))


def class_fractions(counts):
    """Normalize donor–acceptor class counts.

    Parameters
    ----------
    counts : dict
        Class name to count. The classes of one water population are expected.

    Returns
    -------
    dict
        Fractions. They sum to 1 when the counts are positive.

    Raises
    ------
    ValueError
        If the counts are empty or any count is negative.
    """
    if not counts:
        raise ValueError("No donor-acceptor classes were supplied.")
    values = {name: float(count) for name, count in counts.items()}
    if any(count < 0 for count in values.values()):
        raise ValueError("A class count cannot be negative.")
    total = sum(values.values())
    if total <= 0:
        raise ValueError("The class counts sum to zero.")
    return {name: count / total for name, count in values.items()}


__all__ = [
    "directed_hydrogen_bonds",
    "hydrogen_bond_counts",
    "intermittent_retention",
    "class_fractions",
]
