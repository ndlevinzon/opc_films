"""Self and rotational correlation functions.

One unwrapping rule is used: the in-plane displacement is the minimum image
of the oxygen displacement unless the caller has already unwrapped it.
A region mask, when given, is applied to both endpoints of the displacement.
"""

from __future__ import annotations

import numpy as np
from scipy.special import j0


def self_intermediate(displacement_xy, qstar, mask=None):
    """In-plane self-intermediate scattering function.

    Parameters
    ----------
    displacement_xy : array_like, shape (n, 2)
        In-plane oxygen displacement over one lag, in angstroms.
    qstar : float
        Adopted in-plane peak, in inverse angstroms.
    mask : array_like of bool, optional
        Waters that belong to the region at the origin.

    Returns
    -------
    float
        ``mean(J0(q* |dr_xy|))`` over the selected waters.
    """
    displacement = np.asarray(displacement_xy, dtype=float)
    if mask is not None:
        displacement = displacement[np.asarray(mask, dtype=bool)]
    if len(displacement) == 0:
        return np.nan
    distance = np.linalg.norm(displacement, axis=1)
    return float(np.mean(j0(float(qstar) * distance)))


def legendre_correlation(cosine, order, mask=None):
    """Rotational correlation of order 1 or 2.

    Parameters
    ----------
    cosine : array_like
        Dot product of the unit bisector at the origin and at the lag.
    order : int
        ``1`` returns the mean cosine. ``2`` returns ``(3 cos^2 - 1) / 2``.
    mask : array_like of bool, optional
        Waters that belong to the region at the origin.

    Returns
    -------
    float
        The correlation. NaN when the mask is empty.

    Raises
    ------
    ValueError
        If ``order`` is not 1 or 2.
    """
    values = np.asarray(cosine, dtype=float)
    if mask is not None:
        values = values[np.asarray(mask, dtype=bool)]
    if len(values) == 0:
        return np.nan
    if int(order) == 1:
        return float(np.mean(values))
    if int(order) == 2:
        return float(np.mean((3.0 * values ** 2 - 1.0) / 2.0))
    raise ValueError("order must be 1 or 2.")
