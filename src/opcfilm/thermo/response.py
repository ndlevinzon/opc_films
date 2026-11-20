"""Apparent response functions from a per-window table.

``C_V,app`` is the derivative of the potential energy with respect to
temperature, in kcal mol^-1 K^-1. It is configurational. ``alpha`` is the
derivative of the density with respect to temperature, divided by the density.
"""

from __future__ import annotations

import numpy as np


def finite_difference(x, y):
    """Central difference, with a one-sided difference at each end.

    Parameters
    ----------
    x : array_like
        Independent variable. It must be strictly increasing.
    y : array_like
        Dependent variable.

    Returns
    -------
    numpy.ndarray
        ``dy/dx`` at each ``x``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        raise ValueError("A derivative needs at least two points.")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing.")
    slope = np.empty_like(x)
    slope[0] = (y[1] - y[0]) / (x[1] - x[0])
    slope[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    slope[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
    return slope


def heat_capacity(temperature, energy_per_mol):
    """Apparent configurational heat capacity.

    Parameters
    ----------
    temperature : array_like
        Temperature in kelvin.
    energy_per_mol : array_like
        Potential energy per mole, in kcal mol^-1.

    Returns
    -------
    numpy.ndarray
        ``dE/dT`` in kcal mol^-1 K^-1.
    """
    return finite_difference(temperature, energy_per_mol)


def thermal_expansivity(temperature, density):
    """Apparent thermal expansivity.

    Parameters
    ----------
    temperature : array_like
        Temperature in kelvin.
    density : array_like
        Mass density in g cm^-3.

    Returns
    -------
    numpy.ndarray
        ``(1/rho) drho/dT`` in K^-1.
    """
    density = np.asarray(density, dtype=float)
    return finite_difference(temperature, density) / density
