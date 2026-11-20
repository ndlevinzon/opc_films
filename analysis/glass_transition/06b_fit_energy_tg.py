#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.rates import as_physical_rate
from opcfilm.config import CONFIG
from common.trajectory import resolve_n_water


def linear_fit(T, E):

    """Straight-line fit and its coefficient of determination.

    Parameters
    ----------
    T : 
        Temperature in kelvin.
    E : 
        Potential energy per molecule.

    Returns
    -------
    result
        Straight-line fit and its coefficient of determination.
    """
    p = np.polyfit(T, E, 1)

    slope = p[0]
    intercept = p[1]

    pred = slope * T + intercept

    ss_res = np.sum((E - pred) ** 2)
    ss_tot = np.sum((E - np.mean(E)) ** 2)

    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0
        else np.nan
    )

    return slope, intercept, r2


def intersection(mg, bg, ml, bl):

    """Temperature where the glass and liquid lines meet.

    Parameters
    ----------
    mg : 
        Slope of the glass branch of the energy.
    bg : 
        Intercept of the glass branch.
    ml : 
        Slope of the liquid branch of the energy.
    bl : 
        Intercept of the liquid branch.

    Returns
    -------
    result
        Temperature where the glass and liquid lines meet.
    """
    denom = ml - mg

    if abs(denom) < 1.0e-12:
        return np.nan

    return (bg - bl) / denom


def fit_one(
    d,
    glass_min,
    glass_max,
    liquid_min,
    liquid_max
):

    """Glass and liquid lines for one replica and their intersection.

    Parameters
    ----------
    d : 
        Table or per-replica records for one fit.
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
        Glass and liquid lines for one replica and their intersection.
    """
    glass = d[
        (d["Tbin"] >= glass_min)
        &
        (d["Tbin"] <= glass_max)
    ].copy()

    liquid = d[
        (d["Tbin"] >= liquid_min)
        &
        (d["Tbin"] <= liquid_max)
    ].copy()

    if len(glass) < 4 or len(liquid) < 4:
        return None

    mg, bg, r2g = linear_fit(
        glass["Tbin"].to_numpy(),
        glass["E_per_water"].to_numpy()
    )

    ml, bl, r2l = linear_fit(
        liquid["Tbin"].to_numpy(),
        liquid["E_per_water"].to_numpy()
    )

    Tg = intersection(
        mg, bg,
        ml, bl
    )

    return {
        "Tg_K": Tg,

        "glass_slope":
            mg,

        "glass_intercept":
            bg,

        "glass_R2":
            r2g,

        "liquid_slope":
            ml,

        "liquid_intercept":
            bl,

        "liquid_R2":
            r2l,

        "n_glass":
            len(glass),

        "n_liquid":
            len(liquid)
    }


def main():

    """Estimate an energetic glass temperature from two linear branches.

    Returns
    -------
    result
        Estimate an energetic glass temperature from two linear branches.
    """
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-i",
        "--input",
        required=True,
        help="Stage-6 replica-binned CSV"
    )

    ap.add_argument(
        "--n-water",
        type=int,
        default=None,
        help="Optional cross-check against the count read from --topology.",
    )
    ap.add_argument("--topology", required=True, help="Amber prmtop used to count waters.")

    ap.add_argument(
        "--glass-min",
        type=float,
        default=CONFIG.energy.glass_min_K
    )

    ap.add_argument(
        "--glass-max",
        type=float,
        default=CONFIG.energy.glass_max_K
    )

    ap.add_argument(
        "--liquid-min",
        type=float,
        default=CONFIG.energy.liquid_min_K
    )

    ap.add_argument(
        "--liquid-max",
        type=float,
        default=CONFIG.energy.liquid_max_K
    )

    ap.add_argument(
        "--out-prefix",
        default="energy_tg"
    )

    args = ap.parse_args()
    args.n_water = resolve_n_water(args.topology, args.n_water)


    # ========================================================
    # Read Stage 6 replica-level data
    # ========================================================

    df = pd.read_csv(
        args.input
    )

    required = [
        "label",
        "replica",
        "Tbin",
        "EPtot_replica_mean"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing required columns: "
            + ", ".join(missing)
        )


    df["rate_K_per_ns"] = df["label"].map(as_physical_rate)

    df["E_per_water"] = (
        df["EPtot_replica_mean"]
        /
        args.n_water
    )


    # ========================================================
    # PRIMARY fits
    # ========================================================

    rows = []

    for (rate, replica), d in df.groupby(
        [
            "rate_K_per_ns",
            "replica"
        ]
    ):

        d = d.sort_values(
            "Tbin"
        )

        result = fit_one(
            d,
            args.glass_min,
            args.glass_max,
            args.liquid_min,
            args.liquid_max
        )

        if result is None:
            continue

        result.update({
            "rate_K_per_ns":
                rate,

            "replica":
                replica,

            "glass_Tmin":
                args.glass_min,

            "glass_Tmax":
                args.glass_max,

            "liquid_Tmin":
                args.liquid_min,

            "liquid_Tmax":
                args.liquid_max
        })

        rows.append(
            result
        )


    primary = pd.DataFrame(
        rows
    )

    primary = primary.sort_values(
        [
            "rate_K_per_ns",
            "replica"
        ]
    )


    primary_file = (
        args.out_prefix
        + "_by_replica.csv"
    )

    primary.to_csv(
        primary_file,
        index=False
    )


    # ========================================================
    # WINDOW SENSITIVITY
    #
    # Do not rely on one exact choice of fit windows.
    #
    # High branch:
    # 252, 256, 260 -> 300 K
    #
    # Low branch:
    # 120 -> 170, 180, 190 K
    # ========================================================

    liquid_mins = [
        252.0,
        256.0,
        260.0
    ]

    glass_maxs = [
        170.0,
        180.0,
        190.0
    ]

    sens_rows = []

    for (rate, replica), d in df.groupby(
        [
            "rate_K_per_ns",
            "replica"
        ]
    ):

        d = d.sort_values(
            "Tbin"
        )

        for gmax in glass_maxs:

            for lmin in liquid_mins:

                result = fit_one(
                    d,
                    args.glass_min,
                    gmax,
                    lmin,
                    args.liquid_max
                )

                if result is None:
                    continue

                Tg = result["Tg_K"]

                # Reject obviously nonphysical
                # line intersections.
                if not (
                    150.0
                    <= Tg
                    <= 260.0
                ):
                    continue

                sens_rows.append({
                    "rate_K_per_ns":
                        rate,

                    "replica":
                        replica,

                    "glass_Tmin":
                        args.glass_min,

                    "glass_Tmax":
                        gmax,

                    "liquid_Tmin":
                        lmin,

                    "liquid_Tmax":
                        args.liquid_max,

                    **result
                })


    sensitivity = pd.DataFrame(
        sens_rows
    )

    sensitivity_file = (
        args.out_prefix
        + "_window_sensitivity.csv"
    )

    sensitivity.to_csv(
        sensitivity_file,
        index=False
    )


    # ========================================================
    # Replica statistics for PRIMARY Tg
    # ========================================================

    summary = (
        primary
        .groupby(
            "rate_K_per_ns",
            as_index=False
        )
        .agg(
            Tg_energy_mean_K=(
                "Tg_K",
                "mean"
            ),

            Tg_energy_sd_K=(
                "Tg_K",
                "std"
            ),

            Tg_energy_min_K=(
                "Tg_K",
                "min"
            ),

            Tg_energy_max_K=(
                "Tg_K",
                "max"
            ),

            n_replicas=(
                "Tg_K",
                "count"
            ),

            glass_R2_mean=(
                "glass_R2",
                "mean"
            ),

            liquid_R2_mean=(
                "liquid_R2",
                "mean"
            )
        )
    )


    # ========================================================
    # Add fitting-window sensitivity
    # ========================================================

    if len(sensitivity) > 0:

        window_summary = (
            sensitivity
            .groupby(
                "rate_K_per_ns",
                as_index=False
            )
            .agg(
                Tg_window_mean_K=(
                    "Tg_K",
                    "mean"
                ),

                Tg_window_sd_K=(
                    "Tg_K",
                    "std"
                ),

                Tg_window_min_K=(
                    "Tg_K",
                    "min"
                ),

                Tg_window_max_K=(
                    "Tg_K",
                    "max"
                ),

                n_window_fits=(
                    "Tg_K",
                    "count"
                )
            )
        )

        summary = summary.merge(
            window_summary,
            on="rate_K_per_ns",
            how="left"
        )


    summary = summary.sort_values(
        "rate_K_per_ns"
    )

    summary_file = (
        args.out_prefix
        + "_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )


    # ========================================================
    # Tg versus cooling rate
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(7.0, 5.5)
    )

    ax.errorbar(
        summary[
            "rate_K_per_ns"
        ],

        summary[
            "Tg_energy_mean_K"
        ],

        yerr=summary[
            "Tg_energy_sd_K"
        ],

        marker="o",
        linestyle="-",
        capsize=4
    )

    ax.set_xscale(
        "log"
    )

    ax.set_xlabel(
        "Cooling Rate (K/ns)"
    )

    ax.set_ylabel(
        r"$T_g^E$ (K)"
    )

    ax.set_title(
        "Energetic Glass Transition vs Cooling Rate"
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    plot_file = (
        args.out_prefix
        + "_vs_rate.png"
    )

    fig.savefig(
        plot_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


    # ========================================================
    # Print results
    # ========================================================

    print()
    print(
        "=============================================="
    )
    print(
        "PRIMARY ENERGETIC Tg RESULTS"
    )
    print(
        "=============================================="
    )

    print()
    print(
        "Glass fit:"
        f" {args.glass_min:.0f}"
        f"-{args.glass_max:.0f} K"
    )

    print(
        "Liquid fit:"
        f" {args.liquid_min:.0f}"
        f"-{args.liquid_max:.0f} K"
    )

    print()

    print(
        primary[
            [
                "rate_K_per_ns",
                "replica",
                "Tg_K",
                "glass_R2",
                "liquid_R2"
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "=============================================="
    )
    print(
        "ACROSS-REPLICA SUMMARY"
    )
    print(
        "=============================================="
    )
    print()

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Files written:"
    )

    print(
        " ",
        primary_file
    )

    print(
        " ",
        sensitivity_file
    )

    print(
        " ",
        summary_file
    )

    print(
        " ",
        plot_file
    )


if __name__ == "__main__":
    main()
