#!/usr/bin/env python3
"""
Replot V6 defect fraction vs temperature using V6 replica_means.csv.

Expected input:
  replica_means.csv produced by the V6 plot/aggregation workflow.

Conventions:
  - Bulk/interior: solid black
  - AWI: dashed black
  - Temperature axis inverted (hot -> cold left to right)
  - Legend outside plotting window
  - Mean across replicas with light ±SD shading
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from opcfilm.config import CONFIG

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from opcfilm.plotting.conventions import install_figure_conventions
install_figure_conventions()  # temperature axes hot to cold; legends outside panels


def setup_style():
    """Matplotlib defaults for the defect-fraction figure.

    Returns
    -------
    result
        Matplotlib defaults for the defect-fraction figure.
    """
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    })


def save_figure(fig, outbase: Path):
    """Write a PNG and a PDF and close the figure.

    Parameters
    ----------
    fig : 
        Matplotlib figure.
    outbase : 
        Output path without a file extension.

    Returns
    -------
    result
        Write a PNG and a PDF and close the figure.
    """
    outbase.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outbase.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def summarize_region(df, region_name):
    """Mean and standard deviation of the defect fraction in one region.

    Parameters
    ----------
    df : 
        Input table.
    region_name : 
        interior or awi.

    Returns
    -------
    result
        Mean and standard deviation of the defect fraction in one region.
    """
    x = (
        df[df["region"] == region_name]
        .groupby("temp_mid_K")["p_defect"]
        .agg(["mean", "std", "count"])
        .sort_index()
        .reset_index()
    )
    x["std"] = x["std"].fillna(0.0)
    return x


def main():
    """Plot the defect fraction of the bulk and the interface.

    Returns
    -------
    result
        Plot the defect fraction of the bulk and the interface.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--replica-means", type=Path, required=True,
                   help="Path to V6 replica_means.csv")
    p.add_argument("--out", type=Path, required=True,
                   help="Output figure basename, e.g. figures/v6_defect_fraction_0to6A")
    p.add_argument("--rate", type=float, default=4.44,
                   help="Cooling rate to plot (default: 4.44 K/ns)")
    p.add_argument("--awi-inner", type=float, default=0.0,
                   help="Inner AWI bound in Å (default: 0)")
    p.add_argument("--awi-outer", type=float, default=CONFIG.regions.awi_outer_A[1],
                   help="Outer AWI bound in Å (default: 6)")
    p.add_argument("--panel-label", default="(c)",
                   help="Optional panel label")
    p.add_argument("--ylabel", default="Defect fraction",
                   help="Y-axis label")
    p.add_argument("--title", default="",
                   help="Optional title")
    args = p.parse_args()

    setup_style()

    df = pd.read_csv(args.replica_means)

    required = {
        "rate_K_ns", "temp_mid_K", "region",
        "awi_inner_A", "awi_outer_A", "p_defect"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {args.replica_means}: {sorted(missing)}")

    # Filter to chosen rate and AWI band
    df = df[
        np.isclose(df["rate_K_ns"], args.rate) &
        np.isclose(df["awi_inner_A"], args.awi_inner) &
        np.isclose(df["awi_outer_A"], args.awi_outer) &
        df["region"].isin(["interior", "awi"])
    ].copy()

    if df.empty:
        raise RuntimeError(
            "No rows matched the requested rate/AWI band.\n"
            f"rate={args.rate}, awi={args.awi_inner}:{args.awi_outer}"
        )

    bulk = summarize_region(df, "interior")
    awi  = summarize_region(df, "awi")

    fig, ax = plt.subplots(figsize=(4.3, 3.3))
    ax.grid(alpha=0.13)

    # Bulk: solid black
    if not bulk.empty:
        T = bulk["temp_mid_K"].to_numpy(float)
        y = bulk["mean"].to_numpy(float)
        s = bulk["std"].to_numpy(float)
        ax.plot(
            T, y,
            color="black",
            linestyle="-",
            marker="o",
            markersize=2.8,
            linewidth=1.20,
            label="Bulk"
        )
        ax.fill_between(T, y - s, y + s, color="black", alpha=0.08, linewidth=0)

    # AWI: dashed black
    if not awi.empty:
        T = awi["temp_mid_K"].to_numpy(float)
        y = awi["mean"].to_numpy(float)
        s = awi["std"].to_numpy(float)
        ax.plot(
            T, y,
            color="black",
            linestyle="--",
            marker="s",
            markersize=2.8,
            linewidth=1.20,
            label="AWI"
        )
        ax.fill_between(T, y - s, y + s, color="black", alpha=0.05, linewidth=0)

    ax.invert_xaxis()
    ax.set_xlabel("Temperature (K)", fontweight="bold")
    ax.set_ylabel(args.ylabel, fontweight="bold")

    band = f"AWI {args.awi_inner:g}–{args.awi_outer:g} Å"
    ax.set_title((args.title + ", " if args.title else "") + band, pad=8)

    # Match the look of your prior figure
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=2,
        frameon=False,
        fontsize=8,
        handlelength=1.8,
        columnspacing=1.50,
        handletextpad=0.5,
    )

    # Optional panel label
    if args.panel_label:
        ax.text(
            -0.14, 1.06, args.panel_label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="left"
        )

    save_figure(fig, args.out)


if __name__ == "__main__":
    main()
