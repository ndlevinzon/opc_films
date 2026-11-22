#!/usr/bin/env python3
"""
Compare dynamic and energetic glass-arrest temperatures across cooling rates.

Improvements relative to the original comparison script
--------------------------------------------------------
1. Reports Delta T = T_arrest - Tg_dynamic at every cooling rate.
2. Fits and reports both rate-dependence slopes in K per decade:
       T = a + b log10(R)
3. Adds uncertainty to dynamic Tg by bootstrapping the replica-resolved
   equilibrium relaxation data used for the Arrhenius fit (recommended).
4. Propagates dynamic and energetic uncertainties to Delta T.
5. Quantifies agreement of the two rate dependences through the slope
   difference, slope ratio, and relative slope mismatch.
6. Uses a two-panel figure:
       A. Tg_dynamic and energetic arrest vs cooling rate
       B. Delta T vs cooling rate
   This separates agreement in RATE DEPENDENCE from any systematic
   absolute offset between the two operational definitions.

Dynamic bootstrap
-----------------
Supply --dynamic-bootstrap-input with the Stage-4/Stage-5 input table
containing, at minimum:
    T_K (or temperature_K)
    replica
    tau_1e_ns
    status

At each bootstrap iteration, resolved replica tau values are resampled
WITHIN each equilibrium temperature, a new Arrhenius fit is obtained,

    ln(tau) = ln(tau0) + B/T,

and Tg is solved independently for every requested cooling rate using

    tau(Tg) = Tg^2 / (Rcool * B).

This preserves the fact that all dynamic Tg values come from one common
Arrhenius fit and therefore have correlated uncertainties.

If no bootstrap input is supplied, the script will use uncertainty columns
already present in the dynamic CSV when available.  If neither source is
available, the script stops unless --allow-missing-dynamic-uncertainty is
explicitly supplied.

Energetic uncertainty
---------------------
The energetic summary is expected to contain the across-replica mean and SD:
    T_arrest_mean_K
    T_arrest_sd_K

Energetic slope uncertainty is estimated by Monte Carlo sampling each rate
from its across-replica mean and SD.  If an energetic SD is unavailable for
a rate, that rate is held fixed during the Monte Carlo calculation.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.constants import SEEDS
from opcfilm.config import CONFIG


# ============================================================
# Utility functions
# ============================================================

def calc_r2(y, yfit):
    """Coefficient of determination.

    Parameters
    ----------
    y : 
        Dependent variable or profile values.
    yfit : 
        Fitted values of y.

    Returns
    -------
    result
        Coefficient of determination.
    """
    y = np.asarray(y, dtype=float)
    yfit = np.asarray(yfit, dtype=float)

    ss_res = np.sum((y - yfit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    return (
        1.0 - ss_res / ss_tot
        if ss_tot > 0
        else np.nan
    )


def fit_rate_dependence(rate, temperature):
    """Fit:
        T = a + b log10(R)

    Returns intercept a, slope b [K/decade], R^2.

    Parameters
    ----------
    rate : 
        Cooling-rate code or rate in K/ns, as stored in the table.
    temperature : 
        Temperature in kelvin.

    Returns
    -------
    result
        Fit:
    """
    rate = np.asarray(rate, dtype=float)
    temperature = np.asarray(temperature, dtype=float)

    x = np.log10(rate)

    # A line through one cooling rate is undetermined. np.polyfit still
    # returns numbers for it, so the slope is reported as missing instead.
    if np.unique(x[np.isfinite(x)]).size < 2:
        return {
            "intercept_K": float("nan"),
            "slope_K_per_decade": float("nan"),
            "R2": float("nan"),
        }

    b, a = np.polyfit(
        x,
        temperature,
        1,
    )

    pred = a + b * x

    return {
        "intercept_K": float(a),
        "slope_K_per_decade": float(b),
        "R2": float(
            calc_r2(
                temperature,
                pred,
            )
        ),
    }


def percentile_interval(values, ci_percent):
    """Central percentile interval.

    Parameters
    ----------
    values : 
        Numeric observations.
    ci_percent : 
        Confidence level in percent.

    Returns
    -------
    result
        Central percentile interval.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    alpha = (
        100.0 - ci_percent
    ) / 2.0

    lo, hi = np.percentile(
        values,
        [
            alpha,
            100.0 - alpha,
        ],
    )

    return float(lo), float(hi)


def find_column(df, candidates, description, required=True):
    """First candidate column that is present.

    Parameters
    ----------
    df : 
        Input table.
    candidates : 
        Column names to try, in order.
    description : 
        Name of the column used in error messages.
    required : 
        Raise if none of the candidate columns exist.

    Returns
    -------
    result
        First candidate column that is present.
    """
    for c in candidates:
        if c in df.columns:
            return c

    if required:
        raise RuntimeError(
            f"Could not find {description}. "
            f"Tried: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


# ============================================================
# Dynamic Tg table
# ============================================================

def normalize_dynamic_table(path):
    """Dynamic glass temperatures on a common set of column names.

    Parameters
    ----------
    path : 
        Input file or directory.

    Returns
    -------
    result
        Dynamic glass temperatures on a common set of column names.
    """
    dyn = pd.read_csv(
        path
    )

    rate_col = find_column(
        dyn,
        [
            "rate_K_per_ns",
            "cooling_rate_K_per_ns",
            "rate",
        ],
        "dynamic cooling-rate column",
    )

    tg_col = find_column(
        dyn,
        [
            "Tg_dynamic_K",
            "Tg_dyn_K",
            "Tg_K",
        ],
        "dynamic Tg column",
    )

    out = pd.DataFrame({
        "rate_K_per_ns": pd.to_numeric(
            dyn[rate_col],
            errors="coerce",
        ),
        "Tg_dynamic_K": pd.to_numeric(
            dyn[tg_col],
            errors="coerce",
        ),
    })

    # Optional uncertainty columns.
    sd_col = find_column(
        dyn,
        [
            "Tg_dynamic_sd_K",
            "Tg_dyn_sd_K",
            "Tg_sd_K",
            "Tg_dynamic_std_K",
        ],
        "dynamic Tg SD column",
        required=False,
    )

    lo_col = find_column(
        dyn,
        [
            "Tg_dynamic_ci_low_K",
            "Tg_dynamic_ci95_low_K",
            "Tg_dyn_ci_low_K",
            "Tg_ci_low_K",
        ],
        "dynamic Tg CI lower column",
        required=False,
    )

    hi_col = find_column(
        dyn,
        [
            "Tg_dynamic_ci_high_K",
            "Tg_dynamic_ci95_high_K",
            "Tg_dyn_ci_high_K",
            "Tg_ci_high_K",
        ],
        "dynamic Tg CI upper column",
        required=False,
    )

    out["Tg_dynamic_sd_K"] = (
        pd.to_numeric(
            dyn[sd_col],
            errors="coerce",
        )
        if sd_col is not None
        else np.nan
    )

    out["Tg_dynamic_ci_low_K"] = (
        pd.to_numeric(
            dyn[lo_col],
            errors="coerce",
        )
        if lo_col is not None
        else np.nan
    )

    out["Tg_dynamic_ci_high_K"] = (
        pd.to_numeric(
            dyn[hi_col],
            errors="coerce",
        )
        if hi_col is not None
        else np.nan
    )

    out = (
        out.dropna(
            subset=[
                "rate_K_per_ns",
                "Tg_dynamic_K",
            ]
        )
        .sort_values(
            "rate_K_per_ns"
        )
        .reset_index(
            drop=True
        )
    )

    return out


# ============================================================
# Dynamic bootstrap from replica-resolved tau data
# ============================================================

def normalize_tau_bootstrap_input(path_or_glob):
    """Read either:
      1. a combined tau_all.csv / tau_all_whole.csv file, or
      2. a glob such as ./outputs_merged/tau_T*_r*.csv

    If a 'block' column is present, ONLY block == 'whole' is used.

    A status column is preferred, but it is not required for the
    210-230 K bootstrap.  If status/tau_status is absent, any finite,
    positive tau_1e_ns is classified as resolved.  This is appropriate
    for the equilibrium fit temperatures where all three replicas have
    direct 1/e crossings.

    Parameters
    ----------
    path_or_glob : 
        File path or glob pattern.

    Returns
    -------
    result
        Read either:
    """
    import glob as _glob

    matches = sorted(
        _glob.glob(path_or_glob)
    )

    if not matches:
        # Treat as a literal path so the error is more informative.
        matches = [path_or_glob]

    frames = []

    for f in matches:
        if not Path(f).exists():
            raise RuntimeError(
                f"Dynamic bootstrap input not found: {f}"
            )

        d = pd.read_csv(f)
        d["source_file"] = Path(f).name
        frames.append(d)

    raw = pd.concat(
        frames,
        ignore_index=True,
    )

    # Combined tau_all.csv contains whole + block1/block2/block3.
    # Dynamic Tg must use whole-trajectory relaxation values only.
    if "block" in raw.columns:
        raw = raw[
            raw["block"].astype(str).str.lower().eq("whole")
        ].copy()

        if len(raw) == 0:
            raise RuntimeError(
                "A 'block' column was found, but no block == 'whole' rows exist."
            )

    T_col = find_column(
        raw,
        [
            "T_K",
            "temperature_K",
        ],
        "temperature column in dynamic bootstrap input",
    )

    replica_col = find_column(
        raw,
        [
            "replica",
        ],
        "replica column in dynamic bootstrap input",
    )

    tau_col = find_column(
        raw,
        [
            "tau_1e_ns",
        ],
        "tau_1e_ns column in dynamic bootstrap input",
    )

    status_col = find_column(
        raw,
        [
            "status",
            "tau_status",
        ],
        "resolved/censored status column in dynamic bootstrap input",
        required=False,
    )

    tau_numeric = pd.to_numeric(
        raw[tau_col],
        errors="coerce",
    )

    if status_col is not None:
        status = raw[
            status_col
        ].astype(str)
    else:
        # For bootstrap purposes, finite positive direct 1/e crossings
        # are resolved.  Censored rows have NaN tau_1e_ns and therefore
        # will not enter the equilibrium fit.
        status = np.where(
            np.isfinite(tau_numeric)
            & (tau_numeric > 0),
            "resolved",
            "unresolved",
        )

    out = pd.DataFrame({
        "T_K": pd.to_numeric(
            raw[T_col],
            errors="coerce",
        ),
        "replica": raw[
            replica_col
        ],
        "tau_1e_ns": tau_numeric,
        "status": status,
    })

    out = out.dropna(
        subset=[
            "T_K",
            "tau_1e_ns",
        ]
    ).copy()

    return out


def fit_arrhenius_from_temperature_means(T, tau):
    """tau is the geometric-mean tau at each T.
    Fit:
        ln(tau) = ln(tau0) + B/T

    Parameters
    ----------
    T : 
        Temperature in kelvin.
    tau : 
        Relaxation time.

    Returns
    -------
    result
        tau is the geometric-mean tau at each T.
    """
    T = np.asarray(
        T,
        dtype=float,
    )
    tau = np.asarray(
        tau,
        dtype=float,
    )

    x = 1.0 / T
    y = np.log(tau)

    B, ln_tau0 = np.polyfit(
        x,
        y,
        1,
    )

    return float(
        ln_tau0
    ), float(
        B
    )


def solve_dynamic_tg(
    ln_tau0,
    B,
    cooling_rate,
    Tmin=100.0,
    Tmax=300,
):
    """Dynamic glass temperature from one Arrhenius fit and one cooling rate.

    Parameters
    ----------
    ln_tau0 : 
        Natural log of the Arrhenius prefactor.
    B : 
        Arrhenius slope B = Ea/R, in kelvin.
    cooling_rate : 
        Cooling rate in K/ns.
    Tmin : 
        Lower temperature bound in kelvin.
    Tmax : 
        Upper temperature bound in kelvin.

    Returns
    -------
    result
        Dynamic glass temperature from one Arrhenius fit and one cooling rate.
    """
    if not (
        np.isfinite(ln_tau0)
        and np.isfinite(B)
        and B > 0
        and cooling_rate > 0
    ):
        return np.nan

    def f(T):
        """Arrhenius time minus the cooling time.

        Parameters
        ----------
        T : 
            Temperature in kelvin.

        Returns
        -------
        result
            Arrhenius time minus the cooling time.
        """
        tau = math.exp(
            ln_tau0
            + B / T
        )

        criterion = (
            T * T
            / (
                cooling_rate
                * B
            )
        )

        return (
            tau - criterion
        )

    # Scan for a robust sign-changing bracket.
    grid = np.linspace(
        Tmin,
        Tmax,
        1000,
    )

    vals = np.array(
        [
            f(T)
            for T in grid
        ],
        dtype=float,
    )

    for i in range(
        len(grid) - 1
    ):
        if not (
            np.isfinite(
                vals[i]
            )
            and np.isfinite(
                vals[i + 1]
            )
        ):
            continue

        if vals[i] == 0:
            return float(
                grid[i]
            )

        if (
            vals[i]
            * vals[i + 1]
            < 0
        ):
            return float(
                brentq(
                    f,
                    grid[i],
                    grid[i + 1],
                )
            )

    return np.nan


def bootstrap_dynamic_tg(
    tau_input,
    rates,
    fit_Tmin,
    fit_Tmax,
    n_bootstrap,
    ci_percent,
    seed,
):
    """Bootstrap the dynamic glass temperatures, keeping one shared Arrhenius fit.

    Parameters
    ----------
    tau_input : 
        Table or glob of replica relaxation times.
    rates : 
        Cooling rates in K/ns.
    fit_Tmin : 
        Lower temperature of the fit window, in kelvin.
    fit_Tmax : 
        Upper temperature of the fit window, in kelvin.
    n_bootstrap : 
        Number of bootstrap or Monte Carlo samples.
    ci_percent : 
        Confidence level in percent.
    seed : 
        Random-number seed.

    Returns
    -------
    result
        Bootstrap the dynamic glass temperatures, keeping one shared Arrhenius fit.
    """
    d = normalize_tau_bootstrap_input(
        tau_input
    )

    d = d[
        (
            d["status"]
            .str.lower()
            .eq("resolved")
        )
        & (
            d["T_K"]
            >= fit_Tmin
        )
        & (
            d["T_K"]
            <= fit_Tmax
        )
        & (
            d["tau_1e_ns"]
            > 0
        )
    ].copy()

    temperatures = sorted(
        d["T_K"].unique()
    )

    if len(
        temperatures
    ) < 3:
        raise RuntimeError(
            "Dynamic bootstrap requires at least 3 resolved "
            f"temperatures in {fit_Tmin}-{fit_Tmax} K; "
            f"found {temperatures}."
        )

    by_T = {}

    for T in temperatures:
        vals = d.loc[
            d["T_K"] == T,
            "tau_1e_ns",
        ].to_numpy(
            dtype=float
        )

        vals = vals[
            np.isfinite(vals)
            & (vals > 0)
        ]

        if len(vals) < 2:
            raise RuntimeError(
                f"T={T:g} K has only {len(vals)} resolved "
                "replica tau values; bootstrap needs >=2."
            )

        by_T[
            float(T)
        ] = vals

    rates = np.asarray(
        rates,
        dtype=float,
    )

    rng = np.random.default_rng(
        seed
    )

    tg_samples = np.full(
        (
            n_bootstrap,
            len(rates),
        ),
        np.nan,
        dtype=float,
    )

    slope_samples = np.full(
        n_bootstrap,
        np.nan,
        dtype=float,
    )

    B_samples = np.full(
        n_bootstrap,
        np.nan,
        dtype=float,
    )

    ln_tau0_samples = np.full(
        n_bootstrap,
        np.nan,
        dtype=float,
    )

    T_arr = np.array(
        temperatures,
        dtype=float,
    )

    for b in range(
        n_bootstrap
    ):
        tau_geom = []

        for T in temperatures:
            vals = by_T[
                float(T)
            ]

            sampled = rng.choice(
                vals,
                size=len(vals),
                replace=True,
            )

            tau_geom.append(
                math.exp(
                    np.mean(
                        np.log(
                            sampled
                        )
                    )
                )
            )

        try:
            ln_tau0, B = (
                fit_arrhenius_from_temperature_means(
                    T_arr,
                    np.asarray(
                        tau_geom,
                        dtype=float,
                    ),
                )
            )
        except Exception:
            continue

        if not (
            np.isfinite(B)
            and B > 0
        ):
            continue

        values = []

        for j, rate in enumerate(
            rates
        ):
            Tg = solve_dynamic_tg(
                ln_tau0,
                B,
                float(rate),
            )

            tg_samples[
                b,
                j,
            ] = Tg

            values.append(
                Tg
            )

        values = np.asarray(
            values,
            dtype=float,
        )

        if np.all(
            np.isfinite(
                values
            )
        ):
            slope_samples[b] = (
                fit_rate_dependence(
                    rates,
                    values,
                )[
                    "slope_K_per_decade"
                ]
            )

        B_samples[b] = B
        ln_tau0_samples[b] = (
            ln_tau0
        )

    # Remove fully failed iterations.
    valid_rows = np.all(
        np.isfinite(
            tg_samples
        ),
        axis=1,
    )

    tg_samples = (
        tg_samples[
            valid_rows
        ]
    )

    slope_samples = (
        slope_samples[
            valid_rows
        ]
    )

    B_samples = (
        B_samples[
            valid_rows
        ]
    )

    ln_tau0_samples = (
        ln_tau0_samples[
            valid_rows
        ]
    )

    if len(
        tg_samples
    ) < max(
        100,
        int(
            0.5
            * n_bootstrap
        ),
    ):
        raise RuntimeError(
            f"Only {len(tg_samples)}/{n_bootstrap} "
            "dynamic bootstrap iterations were valid."
        )

    rows = []

    for j, rate in enumerate(
        rates
    ):
        vals = tg_samples[
            :,
            j,
        ]

        lo, hi = percentile_interval(
            vals,
            ci_percent,
        )

        rows.append({
            "rate_K_per_ns": float(
                rate
            ),
            "Tg_dynamic_boot_mean_K": float(
                np.mean(vals)
            ),
            "Tg_dynamic_sd_K": float(
                np.std(
                    vals,
                    ddof=1,
                )
            ),
            "Tg_dynamic_ci_low_K": lo,
            "Tg_dynamic_ci_high_K": hi,
        })

    summary = pd.DataFrame(
        rows
    )

    slope_lo, slope_hi = (
        percentile_interval(
            slope_samples,
            ci_percent,
        )
    )

    meta = {
        "seed": int(seed),
        "n_bootstrap_requested": int(
            n_bootstrap
        ),
        "n_bootstrap_valid": int(
            len(tg_samples)
        ),
        "ci_percent": float(
            ci_percent
        ),
        "fit_Tmin_K": float(
            fit_Tmin
        ),
        "fit_Tmax_K": float(
            fit_Tmax
        ),
        "n_equilibrium_temperatures": int(
            len(temperatures)
        ),
        "equilibrium_temperatures_K": (
            ",".join(
                f"{T:g}"
                for T in temperatures
            )
        ),
        "dynamic_slope_boot_mean_K_per_decade": float(
            np.mean(
                slope_samples
            )
        ),
        "dynamic_slope_boot_sd_K_per_decade": float(
            np.std(
                slope_samples,
                ddof=1,
            )
        ),
        "dynamic_slope_boot_ci_low_K_per_decade": slope_lo,
        "dynamic_slope_boot_ci_high_K_per_decade": slope_hi,
    }

    return (
        summary,
        meta,
        slope_samples,
    )


# ============================================================
# Energetic uncertainty / Monte Carlo slope
# ============================================================

def normalize_energetic_table(path):
    """Energetic arrest temperatures on a common set of column names.

    Parameters
    ----------
    path : 
        Input file or directory.

    Returns
    -------
    result
        Energetic arrest temperatures on a common set of column names.
    """
    ene = pd.read_csv(
        path
    )

    rate_col = find_column(
        ene,
        [
            "rate_K_per_ns",
            "cooling_rate_K_per_ns",
            "rate",
        ],
        "energetic cooling-rate column",
    )

    mean_col = find_column(
        ene,
        [
            "T_arrest_mean_K",
            "T_arrest_K",
        ],
        "energetic arrest mean column",
    )

    sd_col = find_column(
        ene,
        [
            "T_arrest_sd_K",
            "T_arrest_std_K",
        ],
        "energetic arrest SD column",
        required=False,
    )

    out = pd.DataFrame({
        "rate_K_per_ns": pd.to_numeric(
            ene[rate_col],
            errors="coerce",
        ),
        "T_arrest_mean_K": pd.to_numeric(
            ene[mean_col],
            errors="coerce",
        ),
        "T_arrest_sd_K": (
            pd.to_numeric(
                ene[sd_col],
                errors="coerce",
            )
            if sd_col is not None
            else np.nan
        ),
    })

    out = (
        out.dropna(
            subset=[
                "rate_K_per_ns",
                "T_arrest_mean_K",
            ]
        )
        .sort_values(
            "rate_K_per_ns"
        )
        .reset_index(
            drop=True
        )
    )

    return out


def monte_carlo_energetic_slope(
    df,
    n_samples,
    ci_percent,
    seed,
):
    """Monte Carlo uncertainty of the energetic arrest slope.

    Parameters
    ----------
    df : 
        Input table.
    n_samples : 
        Number of Monte Carlo samples.
    ci_percent : 
        Confidence level in percent.
    seed : 
        Random-number seed.

    Returns
    -------
    result
        Monte Carlo uncertainty of the energetic arrest slope.
    """
    rate = df[
        "rate_K_per_ns"
    ].to_numpy(
        dtype=float
    )

    mean = df[
        "T_arrest_mean_K"
    ].to_numpy(
        dtype=float
    )

    sd = df[
        "T_arrest_sd_K"
    ].to_numpy(
        dtype=float
    )

    rng = np.random.default_rng(
        seed
    )

    slopes = np.full(
        n_samples,
        np.nan,
        dtype=float,
    )

    # If an SD is unavailable, hold that rate fixed.
    sd_use = np.where(
        np.isfinite(sd)
        & (sd >= 0),
        sd,
        0.0,
    )

    for i in range(
        n_samples
    ):
        y = rng.normal(
            mean,
            sd_use,
        )

        slopes[i] = (
            fit_rate_dependence(
                rate,
                y,
            )[
                "slope_K_per_decade"
            ]
        )

    slopes = slopes[
        np.isfinite(
            slopes
        )
    ]

    lo, hi = percentile_interval(
        slopes,
        ci_percent,
    )

    meta = {
        "seed": int(seed),
        "energetic_slope_mc_mean_K_per_decade": float(
            np.mean(
                slopes
            )
        ),
        "energetic_slope_mc_sd_K_per_decade": float(
            np.std(
                slopes,
                ddof=1,
            )
        ),
        "energetic_slope_mc_ci_low_K_per_decade": lo,
        "energetic_slope_mc_ci_high_K_per_decade": hi,
    }

    return (
        meta,
        slopes,
    )


# ============================================================
# Plot
# ============================================================

def make_plot(
    df,
    dyn_fit,
    ene_fit,
    dyn_slope_sd,
    ene_slope_sd,
    out_prefix,
):
    """Dynamic and energetic arrest temperatures, and their difference, against cooling rate.

    Parameters
    ----------
    df : 
        Input table.
    dyn_fit : 
        Dynamic glass temperatures and their fit against cooling rate.
    ene_fit : 
        Energetic arrest temperatures and their fit against cooling rate.
    dyn_slope_sd : 
        Uncertainty of the dynamic rate-dependence slope.
    ene_slope_sd : 
        Uncertainty of the energetic rate-dependence slope.
    out_prefix : 
        Output path without a file extension.

    Returns
    -------
    result
        Dynamic and energetic arrest temperatures, and their difference, against cooling rate.
    """
    rate = df[
        "rate_K_per_ns"
    ].to_numpy(
        dtype=float
    )

    Tg = df[
        "Tg_dynamic_K"
    ].to_numpy(
        dtype=float
    )

    Tg_sd = df[
        "Tg_dynamic_sd_K"
    ].to_numpy(
        dtype=float
    )

    Ta = df[
        "T_arrest_mean_K"
    ].to_numpy(
        dtype=float
    )

    Ta_sd = df[
        "T_arrest_sd_K"
    ].to_numpy(
        dtype=float
    )

    delta = df[
        "delta_T_K"
    ].to_numpy(
        dtype=float
    )

    delta_sd = df[
        "delta_T_sd_K"
    ].to_numpy(
        dtype=float
    )

    fig, (
        ax,
        axd,
    ) = plt.subplots(
        2,
        1,
        figsize=(
            8.0,
            7.5,
        ),
        sharex=True,
        gridspec_kw={
            "height_ratios": [
                2.8,
                1.25,
            ],
            "hspace": 0.08,
        },
    )

    # --------------------------------------------------------
    # Panel A: two arrest metrics
    # --------------------------------------------------------

    ax.errorbar(
        rate,
        Tg,
        yerr=Tg_sd,
        marker="o",
        markersize=7,
        linewidth=1.6,
        capsize=4,
        label=r"Dynamic $T_g$",
    )

    ax.errorbar(
        rate,
        Ta,
        yerr=Ta_sd,
        marker="s",
        markersize=7,
        linewidth=1.6,
        capsize=4,
        label="Energetic arrest",
    )

    rates_fit = np.logspace(
        np.log10(
            np.min(rate)
        ),
        np.log10(
            np.max(rate)
        ),
        300,
    )

    ax.plot(
        rates_fit,
        dyn_fit[
            "intercept_K"
        ]
        + dyn_fit[
            "slope_K_per_decade"
        ]
        * np.log10(
            rates_fit
        ),
        linestyle="--",
        linewidth=1.20,
    )

    ax.plot(
        rates_fit,
        ene_fit[
            "intercept_K"
        ]
        + ene_fit[
            "slope_K_per_decade"
        ]
        * np.log10(
            rates_fit
        ),
        linestyle="--",
        linewidth=1.20,
    )

    slope_diff = (
        ene_fit[
            "slope_K_per_decade"
        ]
        - dyn_fit[
            "slope_K_per_decade"
        ]
    )

    denom = (
        0.5
        * (
            abs(
                dyn_fit[
                    "slope_K_per_decade"
                ]
            )
            + abs(
                ene_fit[
                    "slope_K_per_decade"
                ]
            )
        )
    )

    rel_mismatch = (
        100.0
        * abs(
            slope_diff
        )
        / denom
        if denom > 0
        else np.nan
    )

    text = (
        "Rate sensitivity\n"
        rf"Dynamic: {dyn_fit['slope_K_per_decade']:.2f}"
    )

    if np.isfinite(
        dyn_slope_sd
    ):
        text += (
            rf" $\pm$ {dyn_slope_sd:.2f}"
        )

    text += (
        " K/decade\n"
        rf"Energetic: {ene_fit['slope_K_per_decade']:.2f}"
    )

    if np.isfinite(
        ene_slope_sd
    ):
        text += (
            rf" $\pm$ {ene_slope_sd:.2f}"
        )

    text += (
        " K/decade\n"
        rf"$\Delta$ slope = {slope_diff:+.2f} K/decade"
        "\n"
        rf"relative mismatch = {rel_mismatch:.1f}%"
    )

    ax.text(
        0.035,
        0.965,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "alpha": 0.84,
            "edgecolor": "0.75",
        },
    )

    ax.set_xscale(
        "log"
    )

    ax.set_ylabel(
        "Arrest temperature (K)"
    )

    ax.set_title(
        "Dynamic and Energetic Measures of Glass Arrest"
    )

    ax.grid(
        alpha=0.20
    )

    ax.legend(
        frameon=False,
        loc="best",
    )

    # --------------------------------------------------------
    # Panel B: absolute offset between criteria
    # --------------------------------------------------------

    axd.errorbar(
        rate,
        delta,
        yerr=delta_sd,
        marker="o",
        linewidth=1.3,
        capsize=4,
    )

    axd.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    mean_delta = float(
        np.mean(
            delta
        )
    )

    sd_delta_rates = (
        float(
            np.std(
                delta,
                ddof=1,
            )
        )
        if len(
            delta
        ) > 1
        else np.nan
    )

    axd.axhline(
        mean_delta,
        linestyle=":",
        linewidth=1.3,
    )

    axd.text(
        0.035,
        0.90,
        (
            rf"mean $\Delta T$ = {mean_delta:+.2f} K"
            + (
                rf"; SD across rates = {sd_delta_rates:.2f} K"
                if np.isfinite(
                    sd_delta_rates
                )
                else ""
            )
        ),
        transform=axd.transAxes,
        va="top",
        ha="left",
        fontsize=9.2,
    )

    axd.set_xlabel(
        "Cooling rate (K/ns)"
    )

    axd.set_ylabel(
        r"$\Delta T$ (K)"
    )

    axd.grid(
        alpha=0.20
    )

    fig.tight_layout()

    png = (
        out_prefix
        + ".png"
    )

    pdf = (
        out_prefix
        + ".pdf"
    )

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return (
        png,
        pdf,
        slope_diff,
        rel_mismatch,
        mean_delta,
        sd_delta_rates,
    )


# ============================================================
# Main
# ============================================================

def main():
    """Compare the rate dependence of the two arrest temperatures.

    Returns
    -------
    result
        Compare the rate dependence of the two arrest temperatures.
    """
    ap = argparse.ArgumentParser(
        description=(
            "Compare Stage-5 dynamic Tg with Stage-6c energetic "
            "arrest, including uncertainties, Delta T, and rate-"
            "dependence slopes in K per decade."
        )
    )

    ap.add_argument(
        "-d",
        "--dynamic",
        required=True,
        help="Dynamic Tg vs cooling-rate CSV.",
    )

    ap.add_argument(
        "-e",
        "--energetic",
        required=True,
        help="Energetic-arrest summary CSV.",
    )

    ap.add_argument(
        "--dynamic-bootstrap-input",
        default=None,
        help=(
            "Recommended: tau_all.csv, tau_all_whole.csv, or a glob "
            "such as ./outputs_merged/tau_T*_r*.csv. If a block column "
            "exists, only block=whole is used automatically. Status is "
            "optional when finite tau_1e_ns values identify resolved points."
        ),
    )

    ap.add_argument(
        "--fit-tmin",
        type=float,
        default=CONFIG.arrhenius.window_K[0],
        help="Lowest equilibrium T used in dynamic Arrhenius bootstrap.",
    )

    ap.add_argument(
        "--fit-tmax",
        type=float,
        default=CONFIG.arrhenius.window_K[1],
        help="Highest equilibrium T used in dynamic Arrhenius bootstrap.",
    )

    ap.add_argument(
        "--n-bootstrap",
        type=int,
        default=5000,
    )

    ap.add_argument(
        "--ci-percent",
        type=float,
        default=95.0,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=SEEDS["energetic_comparison"],
    )

    ap.add_argument(
        "--allow-missing-dynamic-uncertainty",
        action="store_true",
        help=(
            "Allow plotting without dynamic Tg error bars if neither "
            "bootstrap input nor uncertainty columns are available."
        ),
    )

    ap.add_argument(
        "--out-prefix",
        default="./outputs/stage6_dynamic_vs_energetic_improved",
    )

    args = ap.parse_args()

    if not (
        0.0
        < args.ci_percent
        < 100.0
    ):
        raise RuntimeError(
            "--ci-percent must be between 0 and 100."
        )

    if (
        args.n_bootstrap
        < 100
    ):
        raise RuntimeError(
            "--n-bootstrap should be at least 100."
        )

    # --------------------------------------------------------
    # Read central values
    # --------------------------------------------------------

    dyn = normalize_dynamic_table(
        args.dynamic
    )

    ene = normalize_energetic_table(
        args.energetic
    )

    # --------------------------------------------------------
    # Dynamic uncertainty
    # --------------------------------------------------------

    dynamic_boot_meta = None
    dynamic_slope_samples = None

    rates_for_bootstrap = dyn[
        "rate_K_per_ns"
    ].to_numpy(
        dtype=float
    )

    if (
        args.dynamic_bootstrap_input
        is not None
    ):
        (
            boot,
            dynamic_boot_meta,
            dynamic_slope_samples,
        ) = bootstrap_dynamic_tg(
            args.dynamic_bootstrap_input,
            rates_for_bootstrap,
            args.fit_tmin,
            args.fit_tmax,
            args.n_bootstrap,
            args.ci_percent,
            args.seed,
        )

        dyn = dyn.drop(
            columns=[
                "Tg_dynamic_sd_K",
                "Tg_dynamic_ci_low_K",
                "Tg_dynamic_ci_high_K",
            ],
            errors="ignore",
        )

        dyn = pd.merge(
            dyn,
            boot,
            on="rate_K_per_ns",
            how="left",
            validate="one_to_one",
        )

    else:
        has_any_uncertainty = (
            np.any(
                np.isfinite(
                    dyn[
                        "Tg_dynamic_sd_K"
                    ].to_numpy(
                        dtype=float
                    )
                )
            )
            or np.any(
                np.isfinite(
                    dyn[
                        "Tg_dynamic_ci_low_K"
                    ].to_numpy(
                        dtype=float
                    )
                )
            )
        )

        if (
            not has_any_uncertainty
            and not args.allow_missing_dynamic_uncertainty
        ):
            raise RuntimeError(
                "\nDynamic Tg uncertainty is not available.\n\n"
                "Recommended fix: supply\n"
                "  --dynamic-bootstrap-input <replica_resolved_tau.csv>\n\n"
                "The bootstrap input should contain T_K, replica, "
                "tau_1e_ns, and status.\n"
                "This is preferable to inventing an uncertainty for Tg."
            )

    # --------------------------------------------------------
    # Merge dynamic and energetic tables
    # --------------------------------------------------------

    df = pd.merge(
        dyn,
        ene,
        on="rate_K_per_ns",
        how="inner",
        validate="one_to_one",
    )

    df = df.sort_values(
        "rate_K_per_ns"
    ).reset_index(
        drop=True
    )

    if len(
        df
    ) != 5:
        print(
            "WARNING: expected five common cooling rates, "
            f"but found {len(df)}."
        )

    # --------------------------------------------------------
    # If only CI exists but SD does not, keep CI in output but
    # do not silently convert it to an SD.
    # --------------------------------------------------------

    if (
        "Tg_dynamic_sd_K"
        not in df.columns
    ):
        df[
            "Tg_dynamic_sd_K"
        ] = np.nan

    # --------------------------------------------------------
    # Delta T and propagated uncertainty
    # --------------------------------------------------------

    df[
        "delta_T_K"
    ] = (
        df[
            "T_arrest_mean_K"
        ]
        - df[
            "Tg_dynamic_K"
        ]
    )

    df[
        "delta_T_sd_K"
    ] = np.sqrt(
        np.square(
            df[
                "T_arrest_sd_K"
            ].to_numpy(
                dtype=float
            )
        )
        + np.square(
            df[
                "Tg_dynamic_sd_K"
            ].to_numpy(
                dtype=float
            )
        )
    )

    # --------------------------------------------------------
    # Central slope fits
    # --------------------------------------------------------

    rate = df[
        "rate_K_per_ns"
    ].to_numpy(
        dtype=float
    )

    dyn_fit = fit_rate_dependence(
        rate,
        df[
            "Tg_dynamic_K"
        ].to_numpy(
            dtype=float
        ),
    )

    ene_fit = fit_rate_dependence(
        rate,
        df[
            "T_arrest_mean_K"
        ].to_numpy(
            dtype=float
        ),
    )

    # --------------------------------------------------------
    # Energetic slope uncertainty
    # --------------------------------------------------------

    (
        energetic_mc_meta,
        energetic_slope_samples,
    ) = monte_carlo_energetic_slope(
        df[
            [
                "rate_K_per_ns",
                "T_arrest_mean_K",
                "T_arrest_sd_K",
            ]
        ],
        args.n_bootstrap,
        args.ci_percent,
        args.seed + 1,
    )

    dyn_slope_sd = np.nan
    dyn_slope_lo = np.nan
    dyn_slope_hi = np.nan

    if (
        dynamic_slope_samples
        is not None
    ):
        dyn_slope_sd = float(
            np.std(
                dynamic_slope_samples,
                ddof=1,
            )
        )

        (
            dyn_slope_lo,
            dyn_slope_hi,
        ) = percentile_interval(
            dynamic_slope_samples,
            args.ci_percent,
        )

    ene_slope_sd = (
        energetic_mc_meta[
            "energetic_slope_mc_sd_K_per_decade"
        ]
    )

    ene_slope_lo = (
        energetic_mc_meta[
            "energetic_slope_mc_ci_low_K_per_decade"
        ]
    )

    ene_slope_hi = (
        energetic_mc_meta[
            "energetic_slope_mc_ci_high_K_per_decade"
        ]
    )

    # --------------------------------------------------------
    # Slope agreement
    # --------------------------------------------------------

    slope_diff = (
        ene_fit[
            "slope_K_per_decade"
        ]
        - dyn_fit[
            "slope_K_per_decade"
        ]
    )

    slope_ratio = (
        ene_fit[
            "slope_K_per_decade"
        ]
        / dyn_fit[
            "slope_K_per_decade"
        ]
        if dyn_fit[
            "slope_K_per_decade"
        ] != 0
        else np.nan
    )

    denominator = (
        0.5
        * (
            abs(
                dyn_fit[
                    "slope_K_per_decade"
                ]
            )
            + abs(
                ene_fit[
                    "slope_K_per_decade"
                ]
            )
        )
    )

    relative_slope_mismatch_pct = (
        100.0
        * abs(
            slope_diff
        )
        / denominator
        if denominator > 0
        else np.nan
    )

    # Combined slope-difference distribution when dynamic bootstrap exists.
    slope_diff_sd = np.nan
    slope_diff_lo = np.nan
    slope_diff_hi = np.nan

    if (
        dynamic_slope_samples
        is not None
        and len(
            dynamic_slope_samples
        ) > 1
        and len(
            energetic_slope_samples
        ) > 1
    ):
        n = min(
            len(
                dynamic_slope_samples
            ),
            len(
                energetic_slope_samples
            ),
        )

        diff_samples = (
            energetic_slope_samples[
                :n
            ]
            - dynamic_slope_samples[
                :n
            ]
        )

        slope_diff_sd = float(
            np.std(
                diff_samples,
                ddof=1,
            )
        )

        (
            slope_diff_lo,
            slope_diff_hi,
        ) = percentile_interval(
            diff_samples,
            args.ci_percent,
        )

    # --------------------------------------------------------
    # Save comparison table
    # --------------------------------------------------------

    out_csv = (
        args.out_prefix
        + "_comparison.csv"
    )

    df.to_csv(
        out_csv,
        index=False,
    )

    # --------------------------------------------------------
    # Save fit/slope summary
    # --------------------------------------------------------

    fit_rows = [
        {
            "metric": "dynamic_Tg",
            "intercept_K": dyn_fit[
                "intercept_K"
            ],
            "slope_K_per_decade": dyn_fit[
                "slope_K_per_decade"
            ],
            "slope_sd_K_per_decade": dyn_slope_sd,
            "slope_ci_low_K_per_decade": dyn_slope_lo,
            "slope_ci_high_K_per_decade": dyn_slope_hi,
            "R2": dyn_fit[
                "R2"
            ],
        },
        {
            "metric": "energetic_arrest",
            "intercept_K": ene_fit[
                "intercept_K"
            ],
            "slope_K_per_decade": ene_fit[
                "slope_K_per_decade"
            ],
            "slope_sd_K_per_decade": ene_slope_sd,
            "slope_ci_low_K_per_decade": ene_slope_lo,
            "slope_ci_high_K_per_decade": ene_slope_hi,
            "R2": ene_fit[
                "R2"
            ],
        },
        {
            "metric": "energetic_minus_dynamic_slope",
            "intercept_K": np.nan,
            "slope_K_per_decade": slope_diff,
            "slope_sd_K_per_decade": slope_diff_sd,
            "slope_ci_low_K_per_decade": slope_diff_lo,
            "slope_ci_high_K_per_decade": slope_diff_hi,
            "R2": np.nan,
        },
    ]

    fit_summary = pd.DataFrame(
        fit_rows
    )

    fit_summary_file = (
        args.out_prefix
        + "_rate_dependence_fits.csv"
    )

    fit_summary.to_csv(
        fit_summary_file,
        index=False,
    )

    # --------------------------------------------------------
    # Save dynamic bootstrap metadata
    # --------------------------------------------------------

    dynamic_boot_file = None

    if (
        dynamic_boot_meta
        is not None
    ):
        dynamic_boot_file = (
            args.out_prefix
            + "_dynamic_bootstrap_summary.csv"
        )

        pd.DataFrame(
            [
                dynamic_boot_meta
            ]
        ).to_csv(
            dynamic_boot_file,
            index=False,
        )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    (
        out_png,
        out_pdf,
        _,
        _,
        mean_delta,
        sd_delta_rates,
    ) = make_plot(
        df,
        dyn_fit,
        ene_fit,
        dyn_slope_sd,
        ene_slope_sd,
        args.out_prefix,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("=" * 88)
    print("DYNAMIC vs ENERGETIC GLASS ARREST")
    print("=" * 88)

    print()
    print("Rate-by-rate comparison:")
    print(
        df[
            [
                "rate_K_per_ns",
                "Tg_dynamic_K",
                "Tg_dynamic_sd_K",
                "T_arrest_mean_K",
                "T_arrest_sd_K",
                "delta_T_K",
                "delta_T_sd_K",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("Rate dependence: T = a + b log10(R)")
    print()

    print(
        "Dynamic Tg:"
    )
    print(
        f"  slope = "
        f"{dyn_fit['slope_K_per_decade']:.3f} "
        "K/decade"
        + (
            f" +/- {dyn_slope_sd:.3f}"
            if np.isfinite(
                dyn_slope_sd
            )
            else ""
        )
    )
    print(
        f"  R^2   = "
        f"{dyn_fit['R2']:.6f}"
    )

    print()
    print(
        "Energetic arrest:"
    )
    print(
        f"  slope = "
        f"{ene_fit['slope_K_per_decade']:.3f} "
        f"+/- {ene_slope_sd:.3f} K/decade"
    )
    print(
        f"  R^2   = "
        f"{ene_fit['R2']:.6f}"
    )

    print()
    print(
        "Slope agreement:"
    )
    print(
        f"  energetic - dynamic = "
        f"{slope_diff:+.3f} K/decade"
    )
    print(
        f"  slope ratio "
        f"(energetic/dynamic) = "
        f"{slope_ratio:.4f}"
    )
    print(
        f"  relative slope mismatch = "
        f"{relative_slope_mismatch_pct:.2f}%"
    )

    if np.isfinite(
        slope_diff_sd
    ):
        print(
            f"  uncertainty in slope difference = "
            f"{slope_diff_sd:.3f} K/decade"
        )

    print()
    print(
        "Absolute energetic - dynamic offsets:"
    )
    print(
        f"  mean Delta T = "
        f"{mean_delta:+.3f} K"
    )
    print(
        f"  SD of Delta T across rates = "
        f"{sd_delta_rates:.3f} K"
    )

    print()
    print(
        "Interpretation: agreement in the two slopes tests whether "
        "dynamic and energetic arrest have the same cooling-rate "
        "sensitivity. A nonzero Delta T is not by itself a failure "
        "of agreement because the two temperatures use different "
        "operational criteria."
    )

    print()
    print("Wrote:")
    for f in [
        out_csv,
        fit_summary_file,
        dynamic_boot_file,
        out_png,
        out_pdf,
    ]:
        if f is not None:
            print(
                f"  {f}"
            )


if __name__ == "__main__":
    main()
