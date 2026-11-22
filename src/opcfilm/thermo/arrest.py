"""Arrhenius dynamic arrest and the frame spacing that enters it.

The energetic softplus and two-line fits remain in the glass-transition
scripts. Their bounds are the ``energy`` section of the parameter sheet.
This module is the shared Arrhenius solve and the rule that each temperature
keeps its own saved-frame spacing.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from opcfilm.config import R_GAS


def fit_arrhenius(temperature, tau):
    """Linear fit of ln(tau) against 1/T.

    Parameters
    ----------
    temperature : array_like
        Temperature in kelvin.
    tau : array_like
        Relaxation time, in any unit. The slope does not depend on that unit.

    Returns
    -------
    dict
        ``B_K``, ``Ea_kJmol``, ``ln_tau0``, ``tau0``, and ``R2``.
    """
    x = 1.0 / np.asarray(temperature, dtype=float)
    y = np.log(np.asarray(tau, dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    residual = np.sum((y - fitted) ** 2)
    total = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - residual / total if total > 0 else np.nan
    return {
        "ln_tau0": float(intercept),
        "tau0": float(math.exp(intercept)),
        "B_K": float(slope),
        "Ea_kJmol": float(slope * R_GAS / 1000.0),
        "R2": float(r2),
    }


def arrhenius_tau(temperature, fit):
    """Arrhenius relaxation time.

    Parameters
    ----------
    temperature : array_like
        Temperature in kelvin.
    fit : dict
        Result of :func:`fit_arrhenius`.

    Returns
    -------
    numpy.ndarray
        Relaxation time in the same unit as the times that were fit.
    """
    temperature = np.asarray(temperature, dtype=float)
    return np.exp(fit["ln_tau0"] + fit["B_K"] / temperature)


def kinetic_timescale(temperature, cooling_rate, slope):
    """Cooling time ``T^2 / (rate * B)``.

    Parameters
    ----------
    temperature : array_like
        Temperature in kelvin.
    cooling_rate : float
        Cooling rate in K per unit time. The time unit matches ``tau``.
    slope : float
        Arrhenius slope ``B``, in kelvin.

    Returns
    -------
    numpy.ndarray
        Kinetic time at each temperature.
    """
    temperature = np.asarray(temperature, dtype=float)
    return temperature ** 2 / (float(cooling_rate) * float(slope))


def solve_tg_arrhenius(fit, cooling_rate, tmin=100.0, tmax=300):
    """Temperature where the Arrhenius time equals the cooling time.

    Parameters
    ----------
    fit : dict
        Result of :func:`fit_arrhenius`.
    cooling_rate : float
        Cooling rate in K per unit time.
    tmin : float, optional
        Lower bracket, in kelvin.
    tmax : float, optional
        Upper bracket, in kelvin.

    Returns
    -------
    float
        Dynamic arrest temperature in kelvin.
    """

    def residual(temperature):
        """Arrhenius time minus the cooling time.

        Parameters
        ----------
        temperature : float
            Temperature in kelvin.

        Returns
        -------
        float
            Difference of the two times.
        """
        return float(
            arrhenius_tau(temperature, fit)
            - kinetic_timescale(temperature, cooling_rate, fit["B_K"])
        )

    return float(brentq(residual, tmin, tmax))


def spacing_by_temperature(frame):
    """One saved-frame spacing for each temperature.

    Parameters
    ----------
    frame : pandas.DataFrame
        Tau table with ``T_K`` and ``frame_ps_selected``.

    Returns
    -------
    dict
        Temperature in kelvin to spacing in picoseconds.

    Raises
    ------
    RuntimeError
        If the column is missing, or one temperature has more than one spacing.
    """
    if "frame_ps_selected" not in frame.columns:
        raise RuntimeError(
            "The tau table has no frame_ps_selected column."
        )
    spacings = {}
    for temperature, block in frame.groupby("T_K"):
        values = pd.to_numeric(block["frame_ps_selected"], errors="coerce").dropna().unique()
        # Spacings are medians of single-precision NetCDF times, so replicas
        # at one temperature can differ by a few hundredths of a picosecond.
        if len(values) == 0 or np.ptp(values) > max(0.05, 0.005 * float(np.median(values))):
            raise RuntimeError(
                f"frame_ps_selected is not one value at {float(temperature):g} K: {values}."
            )
        spacings[float(temperature)] = float(np.median(values))
    return spacings


def spacing_at(spacings, temperature):
    """Frame spacing of one temperature.

    Parameters
    ----------
    spacings : dict
        Result of :func:`spacing_by_temperature`.
    temperature : float
        Temperature in kelvin.

    Returns
    -------
    float
        Spacing in picoseconds for that temperature's own rows.

    Raises
    ------
    RuntimeError
        If that temperature is absent.
    """
    temperature = float(temperature)
    for key, value in spacings.items():
        if abs(key - temperature) < 1e-6:
            return value
    raise RuntimeError(f"No frame spacing at {temperature:g} K.")
