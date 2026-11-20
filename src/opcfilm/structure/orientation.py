"""Dipole orientation.

``P2`` is ``(3 cos^2 theta - 1) / 2``. ``theta`` is the angle between the
water bisector and the surface normal.
"""

from __future__ import annotations

import numpy as np


def legendre_p2(cos_theta):
    """Second Legendre polynomial of a direction cosine.

    Parameters
    ----------
    cos_theta : array_like
        Cosine of the angle to the surface normal.

    Returns
    -------
    numpy.ndarray
        ``(3 cos^2 theta - 1) / 2``.
    """
    cosine = np.asarray(cos_theta, dtype=float)
    return 0.5 * (3.0 * cosine ** 2 - 1.0)
