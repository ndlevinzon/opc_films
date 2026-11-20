"""One relaxation time and one censoring vocabulary.

A trace that is already below ``1/e`` at the first positive lag is
``fast_censored``. A trace that never crosses is ``slow_censored``. A crossing
is ``resolved``. An empty trace is ``unresolved`` and is not a reported status.
"""

from __future__ import annotations

import math

import numpy as np

from opcfilm.config import CENSORING

LEVEL = math.exp(-1.0)


def classify_tau(t_ps, correlation, level=LEVEL):
    """Locate the ``1/e`` crossing on a log-time axis.

    Parameters
    ----------
    t_ps : array_like
        Time in picoseconds.
    correlation : array_like
        Correlation function at those times.
    level : float, optional
        Threshold, usually ``1/e``.

    Returns
    -------
    status : str
        ``resolved``, ``fast_censored``, ``slow_censored``, or ``unresolved``.
    tau_ps : float
        Crossing time. NaN when the crossing is censored.
    bound_ps : float
        First lag for a fast trace and last lag for a slow trace. NaN when
        the crossing is resolved.
    """
    t_ps = np.asarray(t_ps, dtype=float)
    correlation = np.asarray(correlation, dtype=float)
    finite = np.isfinite(t_ps) & np.isfinite(correlation) & (t_ps > 0)
    times = t_ps[finite]
    values = correlation[finite]
    if len(times) == 0:
        return "unresolved", np.nan, np.nan
    if values[0] <= level:
        return "fast_censored", np.nan, float(times[0])
    for index in range(1, len(times)):
        if values[index] <= level < values[index - 1]:
            left = math.log(times[index - 1])
            right = math.log(times[index])
            if values[index] == values[index - 1]:
                tau = math.sqrt(times[index - 1] * times[index])
            else:
                frac = (level - values[index - 1]) / (values[index] - values[index - 1])
                tau = math.exp(left + frac * (right - left))
            return "resolved", float(tau), np.nan
    return "slow_censored", np.nan, float(times[-1])


def require_status(status):
    """Check a status against the reported vocabulary.

    Parameters
    ----------
    status : str
        Status from :func:`classify_tau`.

    Returns
    -------
    str
        The same status.

    Raises
    ------
    ValueError
        If the status is not one of the three reported values. ``unresolved``
        is rejected here because a table should not contain it.
    """
    if status not in CENSORING:
        raise ValueError(f"status must be one of {CENSORING}.")
    return status


def kww(t, amplitude, tau, beta):
    """Kohlrausch stretched exponential.

    Parameters
    ----------
    t : array_like
        Time, in the same units as ``tau``.
    amplitude : float
        Amplitude.
    tau : float
        Relaxation time.
    beta : float
        Stretching exponent.

    Returns
    -------
    numpy.ndarray
        ``amplitude * exp(-(t / tau) ** beta)``.
    """
    t = np.asarray(t, dtype=float)
    return amplitude * np.exp(-np.power(t / tau, beta))
