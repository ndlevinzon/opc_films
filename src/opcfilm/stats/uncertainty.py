"""Replica uncertainty shared by the plotters.

Curves and bands use the replica mean ± ``band`` standard deviations, with
``band`` equal to 1 unless a plotter is told otherwise. Paired regional
differences use a Student-t 95 percent interval. A standard deviation taken
over molecules or over correlated frames is a distribution width, not an
error bar.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import t

from opcfilm.config import CONFIG, CURVE_BAND_SD


def replica_mean_sd(values):
    """Mean and sample standard deviation across replicas.

    Parameters
    ----------
    values : array_like
        One observation per replica. Non-finite values are dropped.

    Returns
    -------
    mean : float
        Arithmetic mean. ``nan`` when no finite values remain.
    sd : float
        Sample standard deviation. ``nan`` when fewer than two values remain.
    n : int
        Number of finite values.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    count = int(values.size)
    if count == 0:
        return np.nan, np.nan, 0
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if count > 1 else np.nan
    return mean, sd, count


def curve_band(mean, sd, band=None):
    """Lower and upper edges of a replica band.

    Parameters
    ----------
    mean : array_like
        Replica mean.
    sd : array_like
        Sample standard deviation across replicas.
    band : float, optional
        Number of standard deviations drawn on either side of the mean.
        The repository default is one.

    Returns
    -------
    low, high : tuple of numpy.ndarray
        ``mean ± band * sd``.
    """
    if band is None:
        band = CONFIG.uncertainty.curve_band_sd
    mean = np.asarray(mean, dtype=float)
    sd = np.asarray(sd, dtype=float)
    half = float(band) * sd
    return mean - half, mean + half


def curve_caption(band=CURVE_BAND_SD):
    """Caption sentence for a curve band.

    Parameters
    ----------
    band : float, optional
        Standard-deviation multiplier passed to :func:`curve_band`.

    Returns
    -------
    str
        Sentence stating that the band is the replica mean ± ``band`` SD.
    """
    return f"Band: replica mean ± {band:g} SD."


def paired_student_interval(values, confidence=None):
    """Mean and Student-t interval of paired differences.

    Parameters
    ----------
    values : array_like
        One paired difference per replica.
    confidence : float, optional
        Two-sided confidence level.

    Returns
    -------
    mean : float
        Mean of the finite differences.
    sd : float
        Their sample standard deviation.
    halfwidth : float
        Student-t half-width. ``nan`` when fewer than two values are present.
    n : int
        Number of finite differences.
    """
    if confidence is None:
        confidence = CONFIG.uncertainty.paired_confidence
    mean, sd, count = replica_mean_sd(values)
    if count < 2 or not np.isfinite(sd):
        return mean, sd, np.nan, count
    alpha = 1.0 - float(confidence)
    halfwidth = float(t.ppf(1.0 - alpha / 2.0, count - 1) * sd / np.sqrt(count))
    return mean, sd, halfwidth, count


def contrast_caption():
    """Caption sentence for a paired regional difference.

    Returns
    -------
    str
        Sentence stating that the interval is a paired Student-t 95 percent
        interval on bulk minus AWI.
    """
    return "Interval: paired Student-t 95% on bulk minus AWI."
