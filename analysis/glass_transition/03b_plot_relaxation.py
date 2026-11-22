#!/usr/bin/env python3
import argparse
import glob
import math
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from opcfilm.plotting.conventions import install_figure_conventions
install_figure_conventions()  # temperature axes hot to cold; legends outside panels
from scipy.optimize import brentq

LEVEL = math.exp(-1.0)
RGAS_KJ = 0.00831446261815324  # kJ mol^-1 K^-1


def parse_filename(path):
    """Temperature and replica encoded in a tau or Fs file name.

    Parameters
    ----------
    path : 
        Input file or directory.

    Returns
    -------
    result
        Temperature and replica encoded in a tau or Fs file name.
    """
    name = os.path.basename(path)
    m = re.search(r"tau_T(\d+)_r(\d+)\.csv$", name)
    if not m:
        raise ValueError(f"Could not parse T/replica from {name}")
    return int(m.group(1)), int(m.group(2))


def matching_fs_file(tau_file):
    """Fs CSV that belongs to one tau CSV.

    Parameters
    ----------
    tau_file : 
        CSV of relaxation times.

    Returns
    -------
    result
        Fs CSV that belongs to one tau CSV.
    """
    d = os.path.dirname(tau_file)
    n = os.path.basename(tau_file).replace("tau_", "Fs_", 1)
    return os.path.join(d, n)


def get_fs_col(df):
    """Name of the scattering-function column.

    Parameters
    ----------
    df : 
        Input table.

    Returns
    -------
    result
        Name of the scattering-function column.
    """
    if "Fs_parallel" in df.columns:
        return "Fs_parallel"
    if "Fs" in df.columns:
        return "Fs"
    raise RuntimeError("Fs file lacks Fs_parallel/Fs column")


def classify_block(summary_row, fs_block):
    """Mark a 1/e time as resolved, fast-censored, or slow-censored.

    Parameters
    ----------
    summary_row : 
        One row of a relaxation summary.
    fs_block : 
        Scattering-function rows for one block.

    Returns
    -------
    result
        Mark a 1/e time as resolved, fast-censored, or slow-censored.
    """
    fcol = get_fs_col(fs_block)

    d = (
        fs_block.sort_values("lag_ps")
        .reset_index(drop=True)
    )

    # Ignore exact lag zero when testing censoring.
    pos = d[d["lag_ps"] > 0].copy()
    if len(pos) == 0:
        return None

    first_lag_ns = float(pos["lag_ps"].iloc[0]) / 1000.0
    max_lag_ns = float(pos["lag_ps"].max()) / 1000.0
    first_fs = float(pos[fcol].iloc[0])
    min_fs = float(pos[fcol].min())

    tau = summary_row.get("tau_1e_ns", np.nan)
    try:
        tau = float(tau)
    except Exception:
        tau = np.nan

    status = summary_row.get("tau_status", None)
    bound = summary_row.get("tau_bound_ns", np.nan)
    try:
        bound = float(bound)
    except Exception:
        bound = np.nan

    # Prefer explicit status if present.
    if isinstance(status, str):
        status = status.strip()

        if status == "resolved" and np.isfinite(tau):
            return dict(
                status="resolved",
                tau_plot_ns=tau,
                tau_1e_ns=tau,
                first_lag_ns=first_lag_ns,
                max_lag_ns=max_lag_ns,
                first_Fs=first_fs,
                min_Fs=min_fs,
            )

        if status == "fast_censored":
            b = bound if np.isfinite(bound) else first_lag_ns
            return dict(
                status="fast_censored",
                tau_plot_ns=b,
                tau_1e_ns=np.nan,
                first_lag_ns=first_lag_ns,
                max_lag_ns=max_lag_ns,
                first_Fs=first_fs,
                min_Fs=min_fs,
            )

        if status == "slow_censored":
            b = bound if np.isfinite(bound) else max_lag_ns
            return dict(
                status="slow_censored",
                tau_plot_ns=b,
                tau_1e_ns=np.nan,
                first_lag_ns=first_lag_ns,
                max_lag_ns=max_lag_ns,
                first_Fs=first_fs,
                min_Fs=min_fs,
            )

    # Backward-compatible classification for centered-trajectory outputs.
    if first_fs <= LEVEL:
        return dict(
            status="fast_censored",
            tau_plot_ns=first_lag_ns,
            tau_1e_ns=np.nan,
            first_lag_ns=first_lag_ns,
            max_lag_ns=max_lag_ns,
            first_Fs=first_fs,
            min_Fs=min_fs,
        )

    if np.isfinite(tau):
        return dict(
            status="resolved",
            tau_plot_ns=tau,
            tau_1e_ns=tau,
            first_lag_ns=first_lag_ns,
            max_lag_ns=max_lag_ns,
            first_Fs=first_fs,
            min_Fs=min_fs,
        )

    return dict(
        status="slow_censored",
        tau_plot_ns=max_lag_ns,
        tau_1e_ns=np.nan,
        first_lag_ns=first_lag_ns,
        max_lag_ns=max_lag_ns,
        first_Fs=first_fs,
        min_Fs=min_fs,
    )


def load_data(pattern):
    """Load tau summaries and the matching scattering curves.

    Parameters
    ----------
    pattern : 
        Glob pattern.

    Returns
    -------
    result
        Load tau summaries and the matching scattering curves.
    """
    tau_files = sorted(glob.glob(pattern))
    if not tau_files:
        raise RuntimeError(f"No files matched: {pattern}")

    rows = []

    for tau_file in tau_files:
        T, replica = parse_filename(tau_file)
        fs_file = matching_fs_file(tau_file)

        if not os.path.exists(fs_file):
            raise RuntimeError(f"Missing matching Fs file: {fs_file}")

        tau_df = pd.read_csv(tau_file)
        fs_df = pd.read_csv(fs_file)

        common_blocks = sorted(
            set(tau_df["block"]) & set(fs_df["block"])
        )

        for block in common_blocks:
            s = tau_df[tau_df["block"] == block]
            f = fs_df[fs_df["block"] == block]

            if len(s) == 0 or len(f) == 0:
                continue

            result = classify_block(s.iloc[0], f)
            if result is None:
                continue

            result.update(
                temperature_K=T,
                replica=replica,
                block=block,
            )
            rows.append(result)

    return pd.DataFrame(rows)


def summarize_temperatures(df):
    """Geometric-mean relaxation time at each temperature.

    Parameters
    ----------
    df : 
        Input table.

    Returns
    -------
    result
        Geometric-mean relaxation time at each temperature.
    """
    whole = df[df["block"] == "whole"].copy()
    rows = []

    for T in sorted(whole["temperature_K"].unique(), reverse=True):
        d = whole[whole["temperature_K"] == T]

        resolved = d[d["status"] == "resolved"]
        fast = d[d["status"] == "fast_censored"]
        slow = d[d["status"] == "slow_censored"]

        row = dict(
            temperature_K=int(T),
            x_1000_over_T=1000.0 / float(T),
            n_total=len(d),
            n_resolved=len(resolved),
            n_fast=len(fast),
            n_slow=len(slow),
            status="mixed",
            tau_geo_ns=np.nan,
            ln_tau_mean=np.nan,
            ln_tau_sd=np.nan,
            bound_ns=np.nan,
        )

        if len(resolved) >= 2:
            tau = resolved["tau_1e_ns"].to_numpy(dtype=float)
            lntau = np.log(tau)

            row["status"] = "resolved"
            row["tau_geo_ns"] = float(np.exp(np.mean(lntau)))
            row["ln_tau_mean"] = float(np.mean(lntau))
            row["ln_tau_sd"] = float(np.std(lntau, ddof=1))

        elif len(fast) >= 2:
            row["status"] = "fast_censored"
            row["bound_ns"] = float(fast["first_lag_ns"].max())

        elif len(slow) >= 2:
            row["status"] = "slow_censored"
            row["bound_ns"] = float(slow["max_lag_ns"].min())

        elif len(resolved) == 1:
            tau = float(resolved["tau_1e_ns"].iloc[0])
            row["status"] = "resolved_single"
            row["tau_geo_ns"] = tau
            row["ln_tau_mean"] = math.log(tau)

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        "temperature_K", ascending=False
    ).reset_index(drop=True)


def fit_arrhenius(summary, Tmin=210, Tmax=230):
    """Arrhenius fit of the resolved equilibrium relaxation times.

    Parameters
    ----------
    summary : 
        Temperature-level summary table.
    Tmin : 
        Lower temperature bound in kelvin.
    Tmax : 
        Upper temperature bound in kelvin.

    Returns
    -------
    result
        Arrhenius fit of the resolved equilibrium relaxation times.
    """
    d = summary[
        (summary["status"] == "resolved")
        & (summary["temperature_K"] >= Tmin)
        & (summary["temperature_K"] <= Tmax)
        & np.isfinite(summary["ln_tau_mean"])
    ].copy()

    if len(d) < 2:
        raise RuntimeError(
            f"Need >=2 resolved temperatures in {Tmin}-{Tmax} K"
        )

    invT = 1.0 / d["temperature_K"].to_numpy(dtype=float)
    y = d["ln_tau_mean"].to_numpy(dtype=float)

    B, intercept = np.polyfit(invT, y, 1)
    pred = intercept + B * invT

    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return dict(
        Tmin_K=Tmin,
        Tmax_K=Tmax,
        n_temperatures=len(d),
        B_K=float(B),
        ln_tau0_ns=float(intercept),
        tau0_ns=float(math.exp(intercept)),
        Ea_kJ_mol=float(B * RGAS_KJ),
        R2=float(r2),
        fit_temperatures_K=",".join(
            str(int(x)) for x in sorted(d["temperature_K"])
        ),
    )


def predict_ln_tau(T, fit):
    """Natural log of the Arrhenius relaxation time.

    Parameters
    ----------
    T : 
        Temperature in kelvin.
    fit : 
        Arrhenius or energy-fit dictionary.

    Returns
    -------
    result
        Natural log of the Arrhenius relaxation time.
    """
    T = np.asarray(T, dtype=float)
    return fit["ln_tau0_ns"] + fit["B_K"] / T


def add_top_temperature_axis(ax, temperatures):
    """Secondary horizontal axis labeled by the fit temperatures.

    Parameters
    ----------
    ax : 
        Matplotlib axes.
    temperatures : 
        Temperatures in kelvin.

    Returns
    -------
    result
        Secondary horizontal axis labeled by the fit temperatures.
    """
    top = ax.twiny()
    top.set_xlim(ax.get_xlim())

    preferred = [
        T for T in
        [250, 230, 210, 200, 190, 180, 170, 160, 150]
        if T in set(int(x) for x in temperatures)
    ]

    top.set_xticks([1000.0 / T for T in preferred])
    top.set_xticklabels([str(T) for T in preferred])
    top.set_xlabel("Temperature (K)")


def plot_main(summary, fit, out_prefix, fit_Tmin, fit_Tmax):
    """Arrhenius plot of the relaxation time and the dynamic glass temperature.

    Parameters
    ----------
    summary : 
        Temperature-level summary table.
    fit : 
        Arrhenius or energy-fit dictionary.
    out_prefix : 
        Output path without a file extension.
    fit_Tmin : 
        Lower temperature of the fit window, in kelvin.
    fit_Tmax : 
        Upper temperature of the fit window, in kelvin.

    Returns
    -------
    result
        Arrhenius plot of the relaxation time and the dynamic glass temperature.
    """
    resolved = summary[
        summary["status"].isin(["resolved", "resolved_single"])
    ].copy()

    in_fit = (
        (resolved["status"] == "resolved")
        & (resolved["temperature_K"] >= fit_Tmin)
        & (resolved["temperature_K"] <= fit_Tmax)
    )

    included = resolved[in_fit]
    excluded = resolved[~in_fit]

    fast = summary[summary["status"] == "fast_censored"]
    slow = summary[summary["status"] == "slow_censored"]

    fig, (ax, axr) = plt.subplots(
        2, 1,
        figsize=(8.3, 7.4),
        sharex=True,
        gridspec_kw=dict(height_ratios=[3.2, 1.15], hspace=0.08),
    )

    if len(included):
        ax.errorbar(
            included["x_1000_over_T"],
            included["ln_tau_mean"],
            yerr=included["ln_tau_sd"],
            fmt="o",
            markersize=7,
            capsize=4,
            label=f"Resolved, used in fit ({fit_Tmin}-{fit_Tmax} K)",
        )

    if len(excluded):
        err = excluded["ln_tau_sd"].to_numpy(dtype=float)
        err = np.where(np.isfinite(err), err, 0.0)

        ax.errorbar(
            excluded["x_1000_over_T"],
            excluded["ln_tau_mean"],
            yerr=err,
            fmt="o",
            markerfacecolor="none",
            markersize=7,
            capsize=4,
            label="Resolved, excluded from fit",
        )

    if len(fast):
        y = np.log(fast["bound_ns"].to_numpy(dtype=float))
        ax.scatter(
            fast["x_1000_over_T"], y,
            marker="v", s=70,
            label=r"Fast censored: $\tau_\alpha<$ first lag",
        )
        for x, yy in zip(fast["x_1000_over_T"], y):
            ax.annotate(
                "",
                xy=(x, yy - 0.55),
                xytext=(x, yy),
                arrowprops=dict(arrowstyle="->", lw=1.1),
            )

    if len(slow):
        y = np.log(slow["bound_ns"].to_numpy(dtype=float))
        ax.scatter(
            slow["x_1000_over_T"], y,
            marker="^", s=70,
            label=r"Slow censored: $\tau_\alpha>$ analyzed lag",
        )
        for x, yy in zip(slow["x_1000_over_T"], y):
            ax.annotate(
                "",
                xy=(x, yy + 0.55),
                xytext=(x, yy),
                arrowprops=dict(arrowstyle="->", lw=1.1),
            )

    xmin = float(summary["x_1000_over_T"].min())
    xmax = float(summary["x_1000_over_T"].max())

    xline = np.linspace(xmin, xmax, 400)
    Tline = 1000.0 / xline

    ax.plot(
        xline,
        predict_ln_tau(Tline, fit),
        linestyle="--",
        linewidth=1.8,
        label=(
            r"Arrhenius fit: "
            r"$\ln\tau_\alpha=\ln\tau_0+E_a/(RT)$"
        ),
    )

    txt = (
        f"Fit: {fit_Tmin}-{fit_Tmax} K\n"
        f"$E_a$ = {fit['Ea_kJ_mol']:.2f} kJ mol$^{{-1}}$\n"
        f"$R^2$ = {fit['R2']:.4f}"
    )

    ax.text(
        0.035, 0.965, txt,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            alpha=0.82,
            edgecolor="0.75",
        ),
    )

    ax.set_ylabel(r"$\ln \tau_\alpha$  ($\tau_\alpha$ in ns)")
    ax.set_title(
        "Structural Relaxation Across the Supercooled-to-Glassy Regime"
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)

    add_top_temperature_axis(ax, summary["temperature_K"])

    # Lower panel: departure from baseline Arrhenius fit.
    if len(resolved):
        delta = (
            resolved["ln_tau_mean"].to_numpy(dtype=float)
            - predict_ln_tau(
                resolved["temperature_K"].to_numpy(dtype=float),
                fit,
            )
        )
        rd = resolved.copy()
        rd["delta_ln_tau"] = delta

        rfit = rd[
            (rd["status"] == "resolved")
            & (rd["temperature_K"] >= fit_Tmin)
            & (rd["temperature_K"] <= fit_Tmax)
        ]
        rex = rd.drop(rfit.index)

        if len(rfit):
            axr.errorbar(
                rfit["x_1000_over_T"],
                rfit["delta_ln_tau"],
                yerr=rfit["ln_tau_sd"],
                fmt="o",
                markersize=6,
                capsize=3,
            )

        if len(rex):
            err = rex["ln_tau_sd"].to_numpy(dtype=float)
            err = np.where(np.isfinite(err), err, 0.0)
            axr.errorbar(
                rex["x_1000_over_T"],
                rex["delta_ln_tau"],
                yerr=err,
                fmt="o",
                markerfacecolor="none",
                markersize=6,
                capsize=3,
            )

    axr.axhline(0, linestyle="--", linewidth=1.0)
    axr.set_ylabel(r"$\Delta\ln\tau$")
    axr.set_xlabel(r"$1000/T$ (K$^{-1}$)")
    axr.grid(alpha=0.2)

    fig.tight_layout()

    png = out_prefix + "_01_relaxation_arrhenius.png"
    pdf = out_prefix + "_01_relaxation_arrhenius.pdf"

    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    return png, pdf


def main():
    """Fit and plot the dynamic glass temperature from collected relaxation times.

    Returns
    -------
    result
        Fit and plot the dynamic glass temperature from collected relaxation times.
    """
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--summary-glob",
        default="./outputs/tau_T*_r*.csv",
        help="tau_T*_r*.csv glob; matching Fs files must be in same directory",
    )
    ap.add_argument("--fit-Tmin", type=int, default=210)
    ap.add_argument("--fit-Tmax", type=int, default=230)
    ap.add_argument(
        "--out-prefix",
        default="./outputs/stage3_relaxation_improved",
    )

    args = ap.parse_args()

    df = load_data(args.summary_glob)

    classified = args.out_prefix + "_classified.csv"
    df.sort_values(
        ["temperature_K", "replica", "block"],
        ascending=[False, True, True],
    ).to_csv(classified, index=False)

    summary = summarize_temperatures(df)
    summary_file = args.out_prefix + "_temperature_summary.csv"
    summary.to_csv(summary_file, index=False)

    fit = fit_arrhenius(
        summary,
        args.fit_Tmin,
        args.fit_Tmax,
    )
    fit_file = args.out_prefix + "_arrhenius_fit.csv"
    pd.DataFrame([fit]).to_csv(fit_file, index=False)

    p1, p1pdf = plot_main(
        summary,
        fit,
        args.out_prefix,
        args.fit_Tmin,
        args.fit_Tmax,
    )

    print("\nBaseline Arrhenius fit:")
    print(pd.DataFrame([fit]).to_string(index=False))
    print(
        "The Arrhenius-window table is "
        "05_fit_relaxation_tg.py *_arrhenius_window_sensitivity.csv. "
        "It keeps a temperature only when all three replicas are resolved."
    )

    print("\nWrote:")
    for f in [
        classified,
        summary_file,
        fit_file,
        p1,
        p1pdf,
    ]:
        if f is not None:
            print(f"  {f}")


if __name__ == "__main__":
    main()
