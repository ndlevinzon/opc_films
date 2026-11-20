#!/usr/bin/env python3
"""Fit the logarithmic cooling-rate dependence of the dynamic glass transition.

Input tables are the fit summaries written by ``05_fit_relaxation_tg.py``,
one row per cooling rate, or one combined CSV. Required columns are a
cooling rate (``cooling_rate_K_per_ns`` or ``rate_K_ns``) and a glass
temperature (``Tg_K`` or ``Tg_dyn_K``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_points(paths):
    """Collect cooling rate and dynamic glass temperature from fit tables.

    Parameters
    ----------
    paths : sequence of path-like
        One or more CSV files.

    Returns
    -------
    pandas.DataFrame
        Columns ``cooling_rate_K_per_ns`` and ``Tg_dyn_K``, sorted by rate.
    """
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        rate = _column(frame, ("cooling_rate_K_per_ns", "rate_K_ns", "cooling_rate"))
        temperature = _column(frame, ("Tg_dyn_K", "Tg_K", "Tg"))
        frames.append(
            pd.DataFrame(
                {
                    "cooling_rate_K_per_ns": pd.to_numeric(frame[rate]),
                    "Tg_dyn_K": pd.to_numeric(frame[temperature]),
                }
            )
        )
    points = pd.concat(frames, ignore_index=True).dropna()
    return points.sort_values("cooling_rate_K_per_ns")


def _column(frame, candidates):
    """Return the first present column name.

    Parameters
    ----------
    frame : pandas.DataFrame
        Table to search.
    candidates : sequence of str
        Names to try, in order.

    Returns
    -------
    str
        Matching column name.
    """
    for name in candidates:
        if name in frame.columns:
            return name
    raise SystemExit(f"None of {candidates} are columns of the input table.")


def fit_log_rate(rate, temperature):
    """Ordinary-least-squares fit of ``T = a + b log10(rate)``.

    Parameters
    ----------
    rate : array_like
        Cooling rates in K/ns. Every value must be positive.
    temperature : array_like
        Glass temperatures in kelvin.

    Returns
    -------
    intercept, slope, r2 : tuple of float
        Intercept in kelvin, slope in kelvin per decade, and the
        coefficient of determination.
    """
    x = np.log10(np.asarray(rate, dtype=float))
    y = np.asarray(temperature, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    residual = np.sum((y - fitted) ** 2)
    total = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - residual / total if total > 0 else np.nan
    return float(intercept), float(slope), float(r2)


def main(argv=None):
    """Fit and plot dynamic glass temperature against cooling rate.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments, excluding the program name.

    Returns
    -------
    int
        Process status code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tables", nargs="+", type=Path)
    parser.add_argument("-o", "--out-prefix", type=Path, default=Path("tg_dyn_vs_rate"))
    args = parser.parse_args(argv)
    points = load_points(args.tables)
    if len(points) < 2:
        raise SystemExit("Need at least two cooling rates.")
    intercept, slope, r2 = fit_log_rate(points["cooling_rate_K_per_ns"], points["Tg_dyn_K"])
    x = np.log10(points["cooling_rate_K_per_ns"].to_numpy(float))
    points = points.copy()
    points["log10_rate"] = x
    points["Tg_fit_K"] = intercept + slope * x
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    points.to_csv(args.out_prefix.with_suffix(".csv"), index=False)
    figure, axis = plt.subplots(figsize=(7.2, 5.4))
    axis.scatter(x, points["Tg_dyn_K"], s=65, label=r"$T_g^{dyn}$")
    grid = np.linspace(x.min(), x.max(), 200)
    axis.plot(
        grid,
        intercept + slope * grid,
        linewidth=2,
        label=rf"$T_g^{{dyn}} = {intercept:.2f} + {slope:.2f}\log_{{10}} R$",
    )
    axis.set_xlabel(r"$\log_{10} R$ (R in K/ns)")
    axis.set_ylabel(r"$T_g^{dyn}$ (K)")
    axis.set_title("Cooling-rate dependence of the dynamic glass transition")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(args.out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(args.out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    print(f"Intercept = {intercept:.4f} K")
    print(f"Slope     = {slope:.4f} K/decade")
    print(f"R^2       = {r2:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
