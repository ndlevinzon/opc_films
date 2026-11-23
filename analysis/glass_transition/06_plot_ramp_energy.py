#!/usr/bin/env python3
"""
Cooling-rate dependence of OPC-water potential energy.

Revisions relative to the original plotting script
--------------------------------------------------
1. Potential energy is normalized per water molecule.
2. Replica-first statistics are preserved:
       frames -> replica bin means -> across-replica mean/SD.
3. A companion panel shows a representative smooth broken-slope fit,
   using the same energetic-arrest model as Stage 6c.
4. The primary energetic-arrest temperature is T90, defined as the
   temperature at which the fitted dE/dT has departed by 10% from the
   high-temperature liquid slope on cooling.
5. The low-temperature energetic stabilization is quantified as

       Delta E_lowT = E_fastest(lowT) - E_slowest(lowT)

   per water molecule.  A positive value means the slower quench reaches
   a lower (more negative) potential energy.
6. Replica-level fit parameters and across-replica energetic-arrest
   statistics are written to CSV.

Smooth broken-slope model
-------------------------
    E(T) = a + m_g (T - Tref)
           + dm * w * log[1 + exp((T - Tc)/w)]

Low T:
    dE/dT -> m_g

High T:
    dE/dT -> m_g + dm = m_l

Tc:
    midpoint of the slope transition

Primary energetic-arrest temperature:
    T90 = Tc + w ln(0.9/0.1)

which corresponds to a 10% departure from the high-T liquid slope.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from opcfilm.plotting.conventions import install_figure_conventions
install_figure_conventions()  # temperature axes hot to cold; legends outside panels
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.rates import parse_rate as physical_rate
from opcfilm.config import CONFIG
from common.trajectory import resolve_n_water


# ============================================================
# Smooth broken-slope model
# ============================================================

def softplus(x):
    """log(1 + exp(x)), evaluated without overflow.

    Parameters
    ----------
    x : 
        Independent variable.

    Returns
    -------
    result
        log(1 + exp(x)), evaluated without overflow.
    """
    return np.logaddexp(0.0, x)


def sigmoid(x):
    """Logistic function.

    Parameters
    ----------
    x : 
        Independent variable.

    Returns
    -------
    result
        Logistic function.
    """
    return 1.0 / (1.0 + np.exp(-x))


def energy_model(T, a, mg, dm, Tc, w, Tref=210):
    """Smooth join of the glass and liquid energy branches.

    Parameters
    ----------
    T : 
        Temperature in kelvin.
    a : 
        Energy-model intercept.
    mg : 
        Slope of the glass branch of the energy.
    dm : 
        Difference between the liquid and glass energy slopes.
    Tc : 
        Midpoint temperature of the liquid-glass crossover, in kelvin.
    w : 
        Width of the liquid-glass crossover, in kelvin.
    Tref : 
        Reference temperature subtracted inside the energy model, in kelvin.

    Returns
    -------
    result
        Smooth join of the glass and liquid energy branches.
    """
    z = (T - Tc) / w
    return (
        a
        + mg * (T - Tref)
        + dm * w * softplus(z)
    )


def slope_model(T, mg, dm, Tc, w):
    """Temperature derivative of the smooth energy model.

    Parameters
    ----------
    T : 
        Temperature in kelvin.
    mg : 
        Slope of the glass branch of the energy.
    dm : 
        Difference between the liquid and glass energy slopes.
    Tc : 
        Midpoint temperature of the liquid-glass crossover, in kelvin.
    w : 
        Width of the liquid-glass crossover, in kelvin.

    Returns
    -------
    result
        Temperature derivative of the smooth energy model.
    """
    z = (T - Tc) / w
    return mg + dm * sigmoid(z)


def transition_temperature(Tc, w, f):
    """Temperature at which the slope has moved a fraction f of the way from glass to liquid.

    Parameters
    ----------
    Tc : 
        Midpoint temperature of the liquid-glass crossover, in kelvin.
    w : 
        Width of the liquid-glass crossover, in kelvin.
    f : 
        Fraction of the liquid-to-glass slope change used to mark arrest.

    Returns
    -------
    result
        Temperature at which the slope has moved a fraction f of the way from glass to liquid.
    """
    return Tc + w * np.log(f / (1.0 - f))


def linear_slope(d, Tmin, Tmax):
    """Slope of energy versus temperature inside one window.

    Parameters
    ----------
    d : 
        Table or per-replica records for one fit.
    Tmin : 
        Lower temperature bound in kelvin.
    Tmax : 
        Upper temperature bound in kelvin.

    Returns
    -------
    result
        Slope of energy versus temperature inside one window.
    """
    x = d[
        (d["Tbin"] >= Tmin)
        & (d["Tbin"] <= Tmax)
    ]

    if len(x) < 4:
        raise RuntimeError(
            f"Too few points in {Tmin}-{Tmax} K "
            f"for a linear slope estimate."
        )

    p = np.polyfit(
        x["Tbin"].to_numpy(dtype=float),
        x["E_per_water"].to_numpy(dtype=float),
        1,
    )
    return float(p[0])


def fit_replica(
    d,
    fit_min,
    fit_max,
    glass_min,
    glass_max,
    liquid_min,
    liquid_max,
):
    """Fit one replica's energy with the smooth two-branch model.

    Parameters
    ----------
    d : 
        Table or per-replica records for one fit.
    fit_min : 
        Lower temperature used for the representative curve.
    fit_max : 
        Upper temperature used for the representative curve.
    glass_min : 
        Lower temperature of the glass branch, in kelvin.
    glass_max : 
        Upper temperature of the glass branch, in kelvin.
    liquid_min : 
        Lower temperature of the liquid branch, in kelvin.
    liquid_max : 
        Upper temperature of the liquid branch, in kelvin.

    Returns
    -------
    result
        Fit one replica's energy with the smooth two-branch model.
    """
    d = d[
        (d["Tbin"] >= fit_min)
        & (d["Tbin"] <= fit_max)
    ].copy()

    d = d.sort_values("Tbin")

    if len(d) < 12:
        raise RuntimeError(
            f"Only {len(d)} points remain in fit range "
            f"{fit_min}-{fit_max} K."
        )

    T = d["Tbin"].to_numpy(dtype=float)
    E = d["E_per_water"].to_numpy(dtype=float)

    mg0 = linear_slope(
        d,
        glass_min,
        glass_max,
    )
    ml0 = linear_slope(
        d,
        liquid_min,
        liquid_max,
    )

    dm0 = max(
        ml0 - mg0,
        1.0e-5,
    )

    Tc0 = 205.0
    w0 = 12
    Tref = 210

    Eref = np.interp(
        Tref,
        T,
        E,
    )

    a0 = (
        Eref
        - dm0
        * w0
        * softplus(
            (Tref - Tc0) / w0
        )
    )

    p0 = np.array([
        a0,
        mg0,
        dm0,
        Tc0,
        w0,
    ])

    # Energies are kcal/mol/water.
    lower = np.array([
        -50.0,   # a
        0.0,     # glass slope
        0.0,     # slope increase
        150.0,   # Tc
        1.0,     # width
    ])

    upper = np.array([
        0.0,     # a
        0.10,    # glass slope
        0.10,    # slope increase
        260.0,   # Tc
        60.0,    # width
    ])

    def residuals(p):
        """Energy-model residuals passed to the least-squares solver.

        Parameters
        ----------
        p : 
            Argument parser.

        Returns
        -------
        result
            Energy-model residuals passed to the least-squares solver.
        """
        a, mg, dm, Tc, w = p
        pred = energy_model(
            T,
            a,
            mg,
            dm,
            Tc,
            w,
            Tref=Tref,
        )
        return pred - E

    result = least_squares(
        residuals,
        p0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=CONFIG.energy.f_scale_kcal_mol_per_water,
        max_nfev=50000,
    )

    if not result.success:
        raise RuntimeError(
            result.message
        )

    a, mg, dm, Tc, w = result.x
    ml = mg + dm

    pred = energy_model(
        T,
        a,
        mg,
        dm,
        Tc,
        w,
        Tref=Tref,
    )

    resid = E - pred

    ss_res = float(
        np.sum(resid**2)
    )
    ss_tot = float(
        np.sum(
            (E - np.mean(E))**2
        )
    )

    R2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0
        else np.nan
    )

    rmse = float(
        np.sqrt(
            np.mean(resid**2)
        )
    )
    f_scale = CONFIG.energy.f_scale_kcal_mol_per_water
    bound_names = ("a", "glass_slope", "slope_change", "Tc", "width")
    at_bound = [
        name for name, value, lo, hi in zip(
            bound_names, (a, mg, dm, Tc, w), lower, upper
        )
        if value <= lo + 1e-4 or value >= hi - 1e-4
    ]
    if at_bound:
        print(f"WARNING: energetic fit sits on a bound: {', '.join(at_bound)}")
    print(f"RMSE / f_scale = {rmse / f_scale:.3g}")

    T95 = transition_temperature(
        Tc,
        w,
        0.95,
    )
    T90 = transition_temperature(
        Tc,
        w,
        0.90,
    )
    T80 = transition_temperature(
        Tc,
        w,
        0.80,
    )
    T10 = transition_temperature(
        Tc,
        w,
        0.10,
    )

    return {
        "a": float(a),
        "glass_slope": float(mg),
        "liquid_slope": float(ml),
        "slope_change": float(dm),
        "Tc_mid_K": float(Tc),
        "w_K": float(w),
        "T_arrest_95_K": float(T95),
        "T_arrest_90_K": float(T90),
        "T_arrest_80_K": float(T80),
        "transition_width_10_90_K": float(T90 - T10),
        "R2": R2,
        "RMSE_kcalmol_per_water": rmse,
        "f_scale_kcalmol_per_water": f_scale,
        "RMSE_over_f_scale": rmse / f_scale,
        "parameters_at_bound": ",".join(at_bound),
        "n_points": int(len(T)),
    }


# ============================================================
# Input / binning
# ============================================================

def parse_rate(label):
    """Cooling rate encoded in a file or column label.

    Parameters
    ----------
    label : 
        Series label.

    Returns
    -------
    result
        Cooling rate encoded in a file or column label.
    """
    try:
        return physical_rate(label)
    except ValueError as exc:
        raise RuntimeError(
            f"Cooling-rate label '{label}' is not a physical rate. "
            "Use inputs such as 4.44=r1.csv."
        ) from exc


def bin_replicas(args):
    """Mean energy of each replica in temperature bins.

    Parameters
    ----------
    args : 
        Parsed command-line arguments.

    Returns
    -------
    result
        Mean energy of each replica in temperature bins.
    """
    replica_rows = []
    label_order = []

    for item in args.inputs:
        if "=" not in item:
            raise RuntimeError(
                f"Input '{item}' must have rate=csv form."
            )

        label, path = item.split("=", 1)
        rate = parse_rate(label)

        if label not in label_order:
            label_order.append(label)

        df = pd.read_csv(path)

        required = {"EPtot", "temp_K"}
        missing = required - set(df.columns)

        if missing:
            raise RuntimeError(
                f"{path} is missing: "
                + ", ".join(sorted(missing))
            )

        d = df[
            ["temp_K", "EPtot"]
        ].dropna().copy()

        d = d[
            (d["temp_K"] >= args.Tmin)
            & (d["temp_K"] <= args.Tmax)
        ].copy()

        if len(d) == 0:
            print(
                f"WARNING: no points between "
                f"{args.Tmin}-{args.Tmax} K in {path}"
            )
            continue

        d["Tbin"] = (
            np.round(
                d["temp_K"] / args.bin_K
            )
            * args.bin_K
        )

        d["E_per_water"] = (
            d["EPtot"]
            / args.n_water
        )

        b = (
            d.groupby(
                "Tbin",
                as_index=False,
            )
            .agg(
                EPtot_replica_mean=(
                    "EPtot",
                    "mean",
                ),
                EPtot_replica_sd=(
                    "EPtot",
                    "std",
                ),
                E_per_water=(
                    "E_per_water",
                    "mean",
                ),
                E_per_water_within_sd=(
                    "E_per_water",
                    "std",
                ),
                n_frames=(
                    "EPtot",
                    "size",
                ),
            )
        )

        b["label"] = label
        b["rate_K_per_ns"] = rate
        b["replica"] = Path(path).stem

        replica_rows.append(b)

    if not replica_rows:
        raise RuntimeError(
            "No usable input data found."
        )

    replicas = pd.concat(
        replica_rows,
        ignore_index=True,
    )

    return replicas, label_order


def summarize_replicas(replicas, band=1.0):
    """Mean and standard deviation of the replica energies.

    Parameters
    ----------
    replicas : 
        Per-replica fit or energy records.

    Returns
    -------
    result
        Mean and standard deviation of the replica energies.
    """
    summary = (
        replicas
        .groupby(
            [
                "label",
                "rate_K_per_ns",
                "Tbin",
            ],
            as_index=False,
        )
        .agg(
            E_per_water_mean=(
                "E_per_water",
                "mean",
            ),
            E_per_water_sd=(
                "E_per_water",
                "std",
            ),
            EPtot_mean=(
                "EPtot_replica_mean",
                "mean",
            ),
            EPtot_sd=(
                "EPtot_replica_mean",
                "std",
            ),
            n_replicas=(
                "E_per_water",
                "count",
            ),
        )
    )

    summary["E_per_water_band_low"] = (
        summary["E_per_water_mean"]
        - float(band) * summary["E_per_water_sd"]
    )

    summary["E_per_water_band_high"] = (
        summary["E_per_water_mean"]
        + float(band) * summary["E_per_water_sd"]
    )

    return summary


# ============================================================
# Energetic-arrest fits
# ============================================================

def fit_all_replicas(replicas, args):
    """Fit every replica and collect the arrest temperatures.

    Parameters
    ----------
    replicas : 
        Per-replica fit or energy records.
    args : 
        Parsed command-line arguments.

    Returns
    -------
    result
        Fit every replica and collect the arrest temperatures.
    """
    rows = []

    for (
        rate,
        label,
        replica,
    ), d in replicas.groupby(
        [
            "rate_K_per_ns",
            "label",
            "replica",
        ]
    ):
        try:
            result = fit_replica(
                d,
                args.fit_min,
                args.fit_max,
                args.glass_min,
                args.glass_max,
                args.liquid_min,
                args.liquid_max,
            )
        except Exception as exc:
            print(
                f"WARNING: energetic fit failed for "
                f"{label} K/ns, {replica}: {exc}"
            )
            continue

        result.update({
            "rate_K_per_ns": float(rate),
            "label": label,
            "replica": replica,
        })

        rows.append(result)

    if not rows:
        raise RuntimeError(
            "All energetic-arrest fits failed."
        )

    fits = pd.DataFrame(rows)

    fit_summary = (
        fits
        .groupby(
            [
                "rate_K_per_ns",
                "label",
            ],
            as_index=False,
        )
        .agg(
            T_arrest_mean_K=(
                "T_arrest_90_K",
                "mean",
            ),
            T_arrest_sd_K=(
                "T_arrest_90_K",
                "std",
            ),
            T95_mean_K=(
                "T_arrest_95_K",
                "mean",
            ),
            T80_mean_K=(
                "T_arrest_80_K",
                "mean",
            ),
            Tc_mid_mean_K=(
                "Tc_mid_K",
                "mean",
            ),
            transition_width_mean_K=(
                "transition_width_10_90_K",
                "mean",
            ),
            R2_mean=(
                "R2",
                "mean",
            ),
            RMSE_mean=(
                "RMSE_kcalmol_per_water",
                "mean",
            ),
            n_fit_replicas=(
                "replica",
                "count",
            ),
        )
    )

    return fits, fit_summary


# ============================================================
# Low-temperature fast-vs-slow separation
# ============================================================

def low_temperature_separation(
    summary,
    target_T,
    n_water,
):
    """Energy difference between rates at the cold end of the ramp.

    Parameters
    ----------
    summary : 
        Temperature-level summary table.
    target_T : 
        Target temperature in kelvin.
    n_water : 
        Number of water molecules.

    Returns
    -------
    result
        Energy difference between rates at the cold end of the ramp.
    """
    rates = sorted(
        summary["rate_K_per_ns"]
        .dropna()
        .unique()
    )

    if len(rates) < 2:
        raise RuntimeError(
            "Need at least two cooling rates to quantify "
            "low-temperature separation."
        )

    slow_rate = float(min(rates))
    fast_rate = float(max(rates))

    def closest_row(rate):
        """Summary row nearest the requested temperature.

        Parameters
        ----------
        rate : 
            Cooling-rate code or rate in K/ns, as stored in the table.

        Returns
        -------
        result
            Summary row nearest the requested temperature.
        """
        d = summary[
            summary["rate_K_per_ns"]
            == rate
        ].copy()

        if len(d) == 0:
            raise RuntimeError(
                f"No summary data for rate {rate:g}."
            )

        j = (
            d["Tbin"] - target_T
        ).abs().idxmin()

        return d.loc[j]

    slow = closest_row(
        slow_rate
    )
    fast = closest_row(
        fast_rate
    )

    Eslow = float(
        slow["E_per_water_mean"]
    )
    Efast = float(
        fast["E_per_water_mean"]
    )

    delta = (
        Efast - Eslow
    )

    return pd.DataFrame([
        {
            "target_temperature_K": float(target_T),
            "fast_rate_K_per_ns": fast_rate,
            "fast_Tbin_K": float(fast["Tbin"]),
            "fast_E_per_water": Efast,
            "slow_rate_K_per_ns": slow_rate,
            "slow_Tbin_K": float(slow["Tbin"]),
            "slow_E_per_water": Eslow,
            "delta_E_fast_minus_slow_kcalmol_per_water": delta,
            "delta_E_fast_minus_slow_total_kcalmol": (
                delta * n_water
            ),
        }
    ])


# ============================================================
# Representative fitted curve
# ============================================================

def representative_fit_curve(
    fits,
    representative_rate,
    Tmin,
    Tmax,
):
    """Smooth curve drawn for one example rate.

    Parameters
    ----------
    fits : 
        Per-replica fit records.
    representative_rate : 
        Cooling rate drawn as the example curve.
    Tmin : 
        Lower temperature bound in kelvin.
    Tmax : 
        Upper temperature bound in kelvin.

    Returns
    -------
    result
        Smooth curve drawn for one example rate.
    """
    d = fits[
        np.isclose(
            fits["rate_K_per_ns"],
            representative_rate,
        )
    ].copy()

    if len(d) == 0:
        available = sorted(
            fits["rate_K_per_ns"].unique()
        )
        raise RuntimeError(
            f"No energetic fits for representative rate "
            f"{representative_rate:g}. Available: {available}"
        )

    Tgrid = np.linspace(
        Tmin,
        Tmax,
        400,
    )

    pred = []

    for _, row in d.iterrows():
        pred.append(
            energy_model(
                Tgrid,
                row["a"],
                row["glass_slope"],
                row["slope_change"],
                row["Tc_mid_K"],
                row["w_K"],
            )
        )

    pred = np.asarray(
        pred,
        dtype=float,
    )

    mean = np.mean(
        pred,
        axis=0,
    )

    sd = (
        np.std(
            pred,
            axis=0,
            ddof=1,
        )
        if len(pred) > 1
        else np.full_like(
            Tgrid,
            np.nan,
        )
    )

    T90_mean = float(
        d["T_arrest_90_K"].mean()
    )

    T90_sd = (
        float(
            d["T_arrest_90_K"].std(
                ddof=1
            )
        )
        if len(d) > 1
        else np.nan
    )

    return (
        Tgrid,
        mean,
        sd,
        T90_mean,
        T90_sd,
        len(d),
    )


# ============================================================
# Plot
# ============================================================

def make_plot(
    summary,
    fits,
    lowT,
    label_order,
    args,
):
    """Energy and arrest temperature against the cooling schedule.

    Parameters
    ----------
    summary : 
        Temperature-level summary table.
    fits : 
        Per-replica fit records.
    lowT : 
        Temperature at which the low-temperature separation is measured.
    label_order : 
        Order in which series are drawn.
    args : 
        Parsed command-line arguments.

    Returns
    -------
    result
        Energy and arrest temperature against the cooling schedule.
    """
    fig, (
        ax_main,
        ax_fit,
    ) = plt.subplots(
        1,
        2,
        figsize=(12.2, 5.7),
        gridspec_kw={
            "width_ratios": [
                1.65,
                1.0,
            ]
        },
    )

    # --------------------------------------------------------
    # Panel A: all cooling rates, energy per water
    # --------------------------------------------------------

    for label in label_order:
        d = summary[
            summary["label"] == label
        ].copy()

        if len(d) == 0:
            continue

        d = d.sort_values(
            "Tbin"
        )

        x = d[
            "Tbin"
        ].to_numpy(dtype=float)

        y = d[
            "E_per_water_mean"
        ].to_numpy(dtype=float)

        lo = d[
            "E_per_water_band_low"
        ].to_numpy(dtype=float)

        hi = d[
            "E_per_water_band_high"
        ].to_numpy(dtype=float)

        line, = ax_main.plot(
            x,
            y,
            linewidth=1.8,
            label=f"{label}",
        )

        valid = (
            np.isfinite(lo)
            & np.isfinite(hi)
        )

        if np.any(valid):
            ax_main.fill_between(
                x[valid],
                lo[valid],
                hi[valid],
                alpha=0.16,
                color=line.get_color(),
                linewidth=0,
            )

    ax_main.set_xlabel(
        "Temperature (K)"
    )
    ax_main.set_ylabel(
        r"Potential energy per water, "
        r"$E_{\mathrm{pot}}/N_{\mathrm{water}}$ "
        r"(kcal mol$^{-1}$ water$^{-1}$)"
    )
    ax_main.set_xlim(
        args.Tmin,
        args.Tmax,
    )
    ax_main.set_title(
        "A. Cooling-rate-dependent energetic stabilization"
    )
    ax_main.grid(
        alpha=0.20
    )
    ax_main.legend(
        title="Cooling rate (K/ns)",
        frameon=False,
    )

    low = lowT.iloc[0]

    delta = float(
        low[
            "delta_E_fast_minus_slow_kcalmol_per_water"
        ]
    )

    target = float(
        low[
            "target_temperature_K"
        ]
    )

    ax_main.text(
        0.035,
        0.055,
        (
            rf"$\Delta E_{{{target:g}\,\mathrm{{K}}}}"
            rf"=E_{{fast}}-E_{{slow}}"
            rf"={delta:.4f}$ kcal mol$^{{-1}}$ water$^{{-1}}$"
        ),
        transform=ax_main.transAxes,
        va="bottom",
        ha="left",
        fontsize=9.5,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "alpha": 0.82,
            "edgecolor": "0.75",
        },
    )

    # --------------------------------------------------------
    # Panel B: representative transition and smooth fit
    # --------------------------------------------------------

    rep_rate = (
        args.representative_rate
        if args.representative_rate is not None
        else float(
            summary["rate_K_per_ns"].min()
        )
    )

    drep = summary[
        np.isclose(
            summary["rate_K_per_ns"],
            rep_rate,
        )
    ].copy().sort_values(
        "Tbin"
    )

    if len(drep) == 0:
        raise RuntimeError(
            f"No summary curve found for representative rate "
            f"{rep_rate:g} K/ns."
        )

    # Restrict the visible companion panel to the transition region.
    dvis = drep[
        (drep["Tbin"] >= args.transition_plot_min)
        & (drep["Tbin"] <= args.transition_plot_max)
    ].copy()

    x = dvis[
        "Tbin"
    ].to_numpy(dtype=float)
    y = dvis[
        "E_per_water_mean"
    ].to_numpy(dtype=float)
    lo = dvis[
        "E_per_water_band_low"
    ].to_numpy(dtype=float)
    hi = dvis[
        "E_per_water_band_high"
    ].to_numpy(dtype=float)

    line, = ax_fit.plot(
        x,
        y,
        linewidth=1.7,
        label="Replica mean",
    )

    valid = (
        np.isfinite(lo)
        & np.isfinite(hi)
    )

    if np.any(valid):
        ax_fit.fill_between(
            x[valid],
            lo[valid],
            hi[valid],
            alpha=0.16,
            color=line.get_color(),
            linewidth=0,
            label=rf"$\pm {float(args.band_sd):g}$ SD across replicas",
        )

    (
        Tgrid,
        Efit,
        Efit_sd,
        T90_mean,
        T90_sd,
        nfit,
    ) = representative_fit_curve(
        fits,
        rep_rate,
        args.transition_plot_min,
        args.transition_plot_max,
    )

    fit_line, = ax_fit.plot(
        Tgrid,
        Efit,
        linestyle="--",
        linewidth=2.0,
        label="Smooth broken-slope fit",
    )

    if np.all(
        np.isfinite(
            Efit_sd
        )
    ):
        ax_fit.fill_between(
            Tgrid,
            Efit - Efit_sd,
            Efit + Efit_sd,
            alpha=0.12,
            color=fit_line.get_color(),
            linewidth=0,
        )

    ax_fit.axvline(
        T90_mean,
        linestyle=":",
        linewidth=1.7,
        label=r"$T_{\mathrm{arrest}}$ (10% slope departure)",
    )

    if np.isfinite(
        T90_sd
    ):
        ax_fit.axvspan(
            T90_mean - T90_sd,
            T90_mean + T90_sd,
            alpha=0.10,
        )

    arrest_text = (
        rf"$T_{{arrest}}={T90_mean:.1f}$ K"
    )

    if np.isfinite(
        T90_sd
    ):
        arrest_text += (
            rf" $\pm$ {T90_sd:.1f} K"
        )

    arrest_text += (
        f"\n{rep_rate:g} K/ns, "
        f"n={nfit} replicas"
    )

    ax_fit.text(
        0.045,
        0.95,
        arrest_text,
        transform=ax_fit.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "alpha": 0.82,
            "edgecolor": "0.75",
        },
    )

    ax_fit.set_xlim(
        args.transition_plot_min,
        args.transition_plot_max,
    )
    ax_fit.set_xlabel(
        "Temperature (K)"
    )
    ax_fit.set_ylabel(
        r"$E_{\mathrm{pot}}/N_{\mathrm{water}}$ "
        r"(kcal mol$^{-1}$ water$^{-1}$)"
    )
    ax_fit.set_title(
        f"B. Energetic arrest at {rep_rate:g} K/ns"
    )
    ax_fit.grid(
        alpha=0.20
    )
    ax_fit.legend(
        frameon=False,
        fontsize=8.5,
    )

    fig.suptitle(
        args.title,
        y=1.01,
    )

    fig.tight_layout()

    png = (
        args.out_prefix
        + ".png"
    )
    pdf = (
        args.out_prefix
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

    plt.close(fig)

    return (
        png,
        pdf,
        rep_rate,
        T90_mean,
        T90_sd,
    )


# ============================================================
# Main
# ============================================================

def main():
    """Fit and plot the energetic arrest along the cooling ramps.

    Returns
    -------
    result
        Fit and plot the energetic arrest along the cooling ramps.
    """
    ap = argparse.ArgumentParser(
        description=(
            "Compare cooling-ramp potential energy across rates, "
            "normalize per water, quantify low-T stabilization, "
            "and show a representative energetic-arrest fit."
        )
    )

    ap.add_argument(
        "inputs",
        nargs="+",
        help=(
            "rate=csv. Repeat the same rate for independent replicas, e.g. "
            "111=r1.csv 111=r2.csv 111=r3.csv"
        ),
    )

    ap.add_argument(
        "--n-water",
        type=int,
        default=None,
        help="Optional cross-check. It must equal the water count read from --topology.",
    )
    ap.add_argument("--topology", required=True, help="Amber prmtop. N is read from this file.")
    ap.add_argument(
        "--band-sd",
        type=float,
        default=1.0,
        help="Replica band in standard deviations. The default is 1.",
    )

    ap.add_argument(
        "--bin-K",
        type=float,
        default=CONFIG.energy.bin_K,
    )

    ap.add_argument(
        "--Tmin",
        type=float,
        default=CONFIG.energy.report_K,
    )

    ap.add_argument(
        "--Tmax",
        type=float,
        default=CONFIG.energy.fit_max_K,
    )

    ap.add_argument(
        "--lowT-report",
        type=float,
        default=CONFIG.energy.report_K,
        help=(
            "Temperature at which to report fastest-vs-slowest "
            "energy separation (default: 78 K)."
        ),
    )

    # Same energetic-arrest model defaults as Stage 6c.
    ap.add_argument(
        "--fit-min",
        type=float,
        default=CONFIG.energy.fit_min_K,
    )
    ap.add_argument(
        "--fit-max",
        type=float,
        default=CONFIG.energy.fit_max_K,
    )
    ap.add_argument(
        "--glass-min",
        type=float,
        default=CONFIG.energy.glass_min_K,
    )
    ap.add_argument(
        "--glass-max",
        type=float,
        default=CONFIG.energy.glass_max_K,
    )
    ap.add_argument(
        "--liquid-min",
        type=float,
        default=CONFIG.energy.liquid_min_K,
    )
    ap.add_argument(
        "--liquid-max",
        type=float,
        default=CONFIG.energy.liquid_max_K,
    )

    ap.add_argument(
        "--representative-rate",
        type=float,
        default=None,
        help=(
            "Cooling rate for the companion energetic-arrest panel. "
            "Default = slowest supplied rate."
        ),
    )

    ap.add_argument(
        "--transition-plot-min",
        type=float,
        default=150,
    )
    ap.add_argument(
        "--transition-plot-max",
        type=float,
        default=260.0,
    )

    ap.add_argument(
        "--title",
        default=(
            "Cooling-Rate Dependence of Potential Energy "
            "in an OPC Water Slab"
        ),
    )

    ap.add_argument(
        "--out-prefix",
        default="ramp_energy_improved",
    )

    args = ap.parse_args()
    args.n_water = resolve_n_water(args.topology, args.n_water)

    # --------------------------------------------------------
    # 1. Replica-first binning
    # --------------------------------------------------------

    replicas, label_order = bin_replicas(
        args
    )

    replica_file = (
        args.out_prefix
        + "_replica_binned.csv"
    )

    replicas.to_csv(
        replica_file,
        index=False,
    )

    # --------------------------------------------------------
    # 2. Across-replica summary
    # --------------------------------------------------------

    summary = summarize_replicas(
        replicas,
        band=args.band_sd,
    )

    summary_file = (
        args.out_prefix
        + "_mean_sd_per_water.csv"
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    # --------------------------------------------------------
    # 3. Smooth energetic-arrest fits, every replica
    # --------------------------------------------------------

    fits, fit_summary = fit_all_replicas(
        replicas,
        args,
    )

    fit_replica_file = (
        args.out_prefix
        + "_energetic_arrest_by_replica.csv"
    )

    fit_summary_file = (
        args.out_prefix
        + "_energetic_arrest_summary.csv"
    )

    fits.to_csv(
        fit_replica_file,
        index=False,
    )

    fit_summary.to_csv(
        fit_summary_file,
        index=False,
    )

    # --------------------------------------------------------
    # 4. Low-T separation
    # --------------------------------------------------------

    lowT = low_temperature_separation(
        summary,
        args.lowT_report,
        args.n_water,
    )

    lowT_file = (
        args.out_prefix
        + "_lowT_fast_vs_slow.csv"
    )

    lowT.to_csv(
        lowT_file,
        index=False,
    )

    # --------------------------------------------------------
    # 5. Plot
    # --------------------------------------------------------

    (
        png,
        pdf,
        rep_rate,
        rep_T90,
        rep_T90_sd,
    ) = make_plot(
        summary,
        fits,
        lowT,
        label_order,
        args,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    low = lowT.iloc[0]

    print()
    print("=" * 88)
    print("IMPROVED COOLING-RAMP ENERGY ANALYSIS")
    print("=" * 88)

    print()
    print(
        f"Energy normalized by N_water = {args.n_water}"
    )

    print()
    print(
        f"Representative energetic-arrest panel: "
        f"{rep_rate:g} K/ns"
    )

    if np.isfinite(
        rep_T90_sd
    ):
        print(
            f"T_arrest (10% slope departure) = "
            f"{rep_T90:.3f} +/- {rep_T90_sd:.3f} K"
        )
    else:
        print(
            f"T_arrest (10% slope departure) = "
            f"{rep_T90:.3f} K"
        )

    print()
    print(
        f"Low-temperature comparison near "
        f"{args.lowT_report:g} K:"
    )

    print(
        f"  fastest quench: "
        f"{low['fast_rate_K_per_ns']:g} K/ns, "
        f"Tbin={low['fast_Tbin_K']:g} K, "
        f"E/N={low['fast_E_per_water']:.6f} "
        f"kcal mol^-1 water^-1"
    )

    print(
        f"  slowest quench: "
        f"{low['slow_rate_K_per_ns']:g} K/ns, "
        f"Tbin={low['slow_Tbin_K']:g} K, "
        f"E/N={low['slow_E_per_water']:.6f} "
        f"kcal mol^-1 water^-1"
    )

    print(
        f"  Delta E_fast-slow = "
        f"{low['delta_E_fast_minus_slow_kcalmol_per_water']:.6f} "
        f"kcal mol^-1 water^-1"
    )

    print(
        f"  equivalent total-system difference = "
        f"{low['delta_E_fast_minus_slow_total_kcalmol']:.3f} "
        f"kcal/mol"
    )

    print()
    print("Files written:")
    for f in [
        replica_file,
        summary_file,
        fit_replica_file,
        fit_summary_file,
        lowT_file,
        png,
        pdf,
    ]:
        print(f"  {f}")


if __name__ == "__main__":
    main()
