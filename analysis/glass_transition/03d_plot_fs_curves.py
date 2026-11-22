#!/usr/bin/env python3
"""
Plot corrected OPC in-plane Fs(q*,t) curves.

Important correction relative to the old plotting script:
replicas are NOT averaged by exact lag_ps values. Different trajectory
lengths generate slightly different logarithmic lag grids, so exact-lag
grouping can make the apparent mean jump between replicas.

Each replica is instead interpolated in log(time) onto a common lag grid
within the overlap shared by all replicas at a given temperature, and only
then are the replica mean and SD calculated.

The exact point Fs(0)=1 is retained. Default x scaling is symlog so zero lag
can be shown without falsely renormalizing the first positive-lag point.
"""

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
from matplotlib import colormaps
from matplotlib.colors import Normalize

LEVEL = math.exp(-1.0)


def parse_filename(path):
    """Temperature and replica encoded in an Fs file name.

    Parameters
    ----------
    path : 
        Input file or directory.

    Returns
    -------
    result
        Temperature and replica encoded in an Fs file name.
    """
    name = os.path.basename(path)
    m = re.search(r"Fs_T(\d+)_r(\d+)\.csv$", name)
    if not m:
        raise ValueError(
            f"Could not parse temperature/replica from {name}; "
            "expected Fs_T250_r1.csv-style names."
        )
    return int(m.group(1)), int(m.group(2))


def load_fs_data(pattern, block="whole", temps=None):
    """Load scattering curves for the requested block and temperatures.

    Parameters
    ----------
    pattern : 
        Glob pattern.
    block : 
        Trajectory block label, usually whole, block1, block2, or block3.
    temps : 
        Temperatures in kelvin.

    Returns
    -------
    result
        Load scattering curves for the requested block and temperatures.
    """
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"No files matched: {pattern}")

    rows = []
    for f in files:
        T, replica = parse_filename(f)
        if temps is not None and T not in temps:
            continue

        df = pd.read_csv(f)
        required = ["block", "lag_ps", "Fs_parallel"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise RuntimeError(f"{f}: missing required columns {missing}")

        d = df[df["block"] == block].copy()
        if len(d) == 0:
            continue

        d["temperature_K"] = T
        d["replica"] = replica
        d["source_file"] = os.path.basename(f)
        d["lag_ns"] = d["lag_ps"].astype(float) / 1000.0
        rows.append(d)

    if not rows:
        raise RuntimeError("No usable Fs curves after filtering.")

    return pd.concat(rows, ignore_index=True)


def interpolate_replica_to_grid(dr, grid_ns):
    """Interpolate one replica onto a common lag grid.

    Parameters
    ----------
    dr : 
        Per-replica scattering curves at one temperature.
    grid_ns : 
        Common lag grid in nanoseconds.

    Returns
    -------
    result
        Interpolate one replica onto a common lag grid.
    """
    dr = (
        dr.sort_values("lag_ns")
        .drop_duplicates(subset="lag_ns", keep="last")
    )

    pos = dr[(dr["lag_ns"] > 0) & np.isfinite(dr["Fs_parallel"])].copy()
    if len(pos) < 2:
        return np.full(len(grid_ns), np.nan)

    x = pos["lag_ns"].to_numpy(dtype=float)
    y = pos["Fs_parallel"].to_numpy(dtype=float)

    return np.interp(
        np.log(grid_ns),
        np.log(x),
        y,
        left=np.nan,
        right=np.nan,
    )


def build_temperature_summary(df, n_grid=180):
    """Mean and spread of the scattering function at each temperature.

    Parameters
    ----------
    df : 
        Input table.
    n_grid : 
        Number of points on the common lag grid.

    Returns
    -------
    result
        Mean and spread of the scattering function at each temperature.
    """
    out_rows = []

    for T in sorted(df["temperature_K"].unique(), reverse=True):
        dT = df[df["temperature_K"] == T]
        replicas = sorted(dT["replica"].unique())

        mins = []
        maxs = []
        usable_replicas = []

        for r in replicas:
            dr = dT[dT["replica"] == r]
            pos = dr[(dr["lag_ns"] > 0) & np.isfinite(dr["Fs_parallel"])]
            if len(pos) < 2:
                continue
            mins.append(float(pos["lag_ns"].min()))
            maxs.append(float(pos["lag_ns"].max()))
            usable_replicas.append(r)

        if not usable_replicas:
            continue

        tmin = max(mins)
        tmax = min(maxs)
        if not (tmax > tmin > 0):
            raise RuntimeError(
                f"T={T}: replicas have no common positive-lag interval."
            )

        grid = np.geomspace(tmin, tmax, n_grid)
        values = []

        for r in usable_replicas:
            dr = dT[dT["replica"] == r]
            values.append(interpolate_replica_to_grid(dr, grid))

        values = np.asarray(values, dtype=float)

        # Exact normalization point.
        out_rows.append({
            "temperature_K": int(T),
            "lag_ns": 0.0,
            "lag_ps": 0.0,
            "Fs_mean": 1.0,
            "Fs_sd": 0.0,
            "n_replicas": int(len(usable_replicas)),
        })

        for j, t in enumerate(grid):
            v = values[:, j]
            v = v[np.isfinite(v)]
            if len(v) == 0:
                continue
            mean = float(np.mean(v))
            sd = float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
            out_rows.append({
                "temperature_K": int(T),
                "lag_ns": float(t),
                "lag_ps": float(t * 1000.0),
                "Fs_mean": mean,
                "Fs_sd": sd,
                "n_replicas": int(len(v)),
            })

    return (
        pd.DataFrame(out_rows)
        .sort_values(["temperature_K", "lag_ns"], ascending=[False, True])
        .reset_index(drop=True)
    )


def plot_fs_curves(
    df,
    summary,
    out_prefix,
    title,
    show_replicas=True,
    xscale="symlog",
    linthresh_ns=None,
):
    """Draw the scattering function, colored by temperature.

    Parameters
    ----------
    df : 
        Input table.
    summary : 
        Temperature-level summary table.
    out_prefix : 
        Output path without a file extension.
    title : 
        Figure title.
    show_replicas : 
        Draw the individual replicas as well as the mean.
    xscale : 
        Matplotlib scale name.
    linthresh_ns : 
        Linear threshold of a symlog time axis, in nanoseconds.

    Returns
    -------
    result
        Draw the scattering function, colored by temperature.
    """
    temperatures = sorted(df["temperature_K"].unique(), reverse=True)
    cmap = colormaps["RdYlBu_r"]
    norm = Normalize(vmin=min(temperatures), vmax=max(temperatures))

    fig, ax = plt.subplots(figsize=(8.2, 6.0))

    positive_lags = df.loc[df["lag_ns"] > 0, "lag_ns"].to_numpy(dtype=float)
    if linthresh_ns is None and len(positive_lags) > 0:
        linthresh_ns = float(np.nanmin(positive_lags))

    for T in temperatures:
        color = cmap(norm(T))
        dT = df[df["temperature_K"] == T].copy()
        sT = summary[summary["temperature_K"] == T].sort_values("lag_ns")

        if show_replicas:
            for replica in sorted(dT["replica"].unique()):
                dr = (
                    dT[dT["replica"] == replica]
                    .sort_values("lag_ns")
                    .drop_duplicates(subset="lag_ns", keep="last")
                )
                if xscale == "log":
                    dr = dr[dr["lag_ns"] > 0]
                ax.plot(
                    dr["lag_ns"],
                    dr["Fs_parallel"],
                    color=color,
                    linewidth=0.7,
                    alpha=0.18,
                )

        splot = sT[sT["lag_ns"] > 0].copy() if xscale == "log" else sT.copy()

        ax.plot(
            splot["lag_ns"],
            splot["Fs_mean"],
            color=color,
            linewidth=2.2,
            label=f"{T} K",
        )

        valid = (
            np.isfinite(splot["Fs_sd"].to_numpy())
            & (splot["n_replicas"].to_numpy() >= 2)
        )
        if np.any(valid):
            x = splot["lag_ns"].to_numpy()[valid]
            y = splot["Fs_mean"].to_numpy()[valid]
            sd = splot["Fs_sd"].to_numpy()[valid]
            ax.fill_between(
                x,
                y - sd,
                y + sd,
                alpha=0.10,
                color=color,
                linewidth=0,
            )

    ax.axhline(
        LEVEL,
        linestyle="--",
        linewidth=1.20,
        label=r"$F_s(q^*,t)=e^{-1}$",
        color="black",
    )

    if xscale == "symlog":
        ax.set_xscale("symlog", linthresh=linthresh_ns, linscale=1.0)
    elif xscale == "log":
        ax.set_xscale("log")
    elif xscale == "linear":
        ax.set_xscale("linear")
    else:
        raise ValueError("xscale must be symlog, log, or linear")

    ax.set_xlabel(r"Lag time, $t$ (ns)")
    ax.set_ylabel(r"$F_s^{\parallel}(q^*,t)$")
    ax.set_title(title)
    ax.set_ylim(-0.08, 1.04)
    ax.grid(alpha=0.20)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        ncol=1,
        fontsize=10,
    )
    fig.tight_layout()

    png = out_prefix + ".png"
    pdf = out_prefix + ".pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main():
    """Plot the in-plane scattering functions.

    Returns
    -------
    result
        Plot the in-plane scattering functions.
    """
    ap = argparse.ArgumentParser(
        description=(
            "Plot corrected OPC in-plane Fs curves after interpolation of "
            "replicas onto a common lag grid."
        )
    )
    ap.add_argument("--fs-glob", default="./outputs_fs_corrected/Fs_T*_r*.csv")
    ap.add_argument(
        "--block",
        default="whole",
        choices=["whole", "block1", "block2", "block3"],
    )
    ap.add_argument("--temps", nargs="*", type=int, default=None)
    ap.add_argument("--no-replicas", action="store_true")
    ap.add_argument("--n-grid", type=int, default=180)
    ap.add_argument(
        "--xscale",
        choices=["symlog", "log", "linear"],
        default="symlog",
        help=(
            "symlog displays the exact t=0, Fs=1 point; log omits t=0 "
            "and gives a conventional log-time plot."
        ),
    )
    ap.add_argument("--linthresh-ns", type=float, default=None)
    ap.add_argument(
        "--title",
        default="In-Plane Self-Intermediate Scattering Function of OPC Water",
    )
    ap.add_argument(
        "--out-prefix",
        default="./outputs_fs_corrected/stage3_fs_curves_corrected",
    )
    args = ap.parse_args()

    df = load_fs_data(args.fs_glob, block=args.block, temps=args.temps)
    summary = build_temperature_summary(df, n_grid=args.n_grid)

    summary_file = args.out_prefix + "_replica_interpolated_summary.csv"
    summary.to_csv(summary_file, index=False)

    png, pdf = plot_fs_curves(
        df,
        summary,
        args.out_prefix,
        args.title,
        show_replicas=(not args.no_replicas),
        xscale=args.xscale,
        linthresh_ns=args.linthresh_ns,
    )

    print()
    print("Wrote:")
    print(f"  {summary_file}")
    print(f"  {png}")
    print(f"  {pdf}")


if __name__ == "__main__":
    main()
