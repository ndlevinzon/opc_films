"""Bulk-minus-interface differences.

Every regional contrast in this repository is bulk minus AWI. A negative
value is an excess at the interface.
"""

from __future__ import annotations

import numpy as np

from opcfilm.config import CONFIG


def bulk_minus_awi(bulk, awi):
    """Subtract the interface value from the bulk value.

    Parameters
    ----------
    bulk : array_like
        Bulk observation. Paired with ``awi`` element by element.
    awi : array_like
        Interface observation.

    Returns
    -------
    numpy.ndarray
        ``bulk - awi``. The name of this contrast is
        ``common.constants.CONTRAST``.
    """
    if CONFIG.uncertainty.contrast != "bulk_minus_awi":
        raise RuntimeError("The regional contrast is no longer bulk minus AWI.")
    return np.asarray(bulk, dtype=float) - np.asarray(awi, dtype=float)
