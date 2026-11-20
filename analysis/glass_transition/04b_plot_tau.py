#!/usr/bin/env python3

import argparse
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LEVEL = math.exp(-1.0)


# ============================================================
# GEOMETRIC MEAN
# ============================================================

def geometric_mean(x):

    """Geometric mean of the positive finite values.

    Parameters
    ----------
    x : 
        Independent variable.

    Returns
    -------
    result
        Geometric mean of the positive finite values.
    """
    x = np.asarray(x, dtype=float)

    x = x[
        np.isfinite(x)
        &
        (x > 0)
    ]

    if len(x) == 0:
        return np.nan

    return float(
        np.exp(
            np.mean(
                np.log(x)
            )
        )
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data(path):

    """Read a collected relaxation table and keep resolved values separate from bounds.

    Parameters
    ----------
    path : 
        Input file or directory.

    Returns
    -------
    result
        Read a collected relaxation table and keep resolved values separate from bounds.
    """
    df = pd.read_csv(path)

    required = [
        "T_K",
        "replica",
        "block",
        "tau_1e_ns"
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

    for c in [
        "T_K",
        "replica",
        "tau_1e_ns"
    ]:

        df[c] = pd.to_numeric(
            df[c],
            errors="coerce"
        )

    if "tau_plot_ns" in df.columns:

        df["tau_plot_ns"] = pd.to_numeric(
            df["tau_plot_ns"],
            errors="coerce"
        )

    return df


# ============================================================
# PLOT 1
#
# WHOLE-TRAJECTORY CENSORED RELAXATION
#
# Resolved:
#     actual tau_alpha
#
# Fast censored:
#     tau_alpha < first saved lag
#
# Slow censored:
#     tau_alpha > maximum analyzed lag
# ============================================================

def plot_censored(df, out_prefix):

    """Relaxation time against temperature, with censoring drawn as bounds.

    Parameters
    ----------
    df : 
        Input table.
    out_prefix : 
        Output path without a file extension.

    Returns
    -------
    result
        Relaxation time against temperature, with censoring drawn as bounds.
    """
    if "status" not in df.columns:

        raise RuntimeError(
            "Stage-4 file has no 'status' column. "
            "Run Stage 4 with --classified."
        )

    if "tau_plot_ns" not in df.columns:

        raise RuntimeError(
            "Stage-4 file has no tau_plot_ns column."
        )

    whole = df[
        df["block"] == "whole"
    ].copy()

    fig, ax = plt.subplots(
        figsize=(7.6, 5.6)
    )

    # --------------------------------------------------------
    # Resolved individual replicas
    # --------------------------------------------------------

    resolved = whole[
        whole["status"] == "resolved"
    ].copy()

    ax.scatter(
        1000.0 / resolved["T_K"],
        np.log10(
            resolved["tau_1e_ns"]
        ),
        s=42,
        alpha=0.45,
        label="Resolved replicas"
    )

    # --------------------------------------------------------
    # Temperature-level geometric means
    # --------------------------------------------------------

    mean_rows = []

    for T, d in resolved.groupby("T_K"):

        vals = d[
            "tau_1e_ns"
        ].to_numpy(dtype=float)

        if len(vals) == 0:
            continue

        logs = np.log10(vals)

        mean_rows.append({

            "T_K": T,

            "tau_geom":
                geometric_mean(vals),

            "log_sd":
                np.std(
                    logs,
                    ddof=1
                )
                if len(logs) > 1
                else 0.0
        })

    means = pd.DataFrame(mean_rows)

    if len(means):

        means = means.sort_values(
            "T_K",
            ascending=False
        )

        ax.errorbar(
            1000.0 / means["T_K"],
            np.log10(
                means["tau_geom"]
            ),
            yerr=means["log_sd"],
            fmt="o-",
            markersize=7,
            capsize=4,
            linewidth=1.8,
            label="Resolved replica mean"
        )

    # --------------------------------------------------------
    # Fast censored
    # --------------------------------------------------------

    fast = whole[
        whole["status"] == "fast_censored"
    ].copy()

    if len(fast):

        ax.scatter(
            1000.0 / fast["T_K"],
            np.log10(
                fast["tau_plot_ns"]
            ),
            marker="v",
            s=65,
            label=r"$\tau_\alpha <$ first saved lag"
        )

        for _, row in fast.iterrows():

            x = 1000.0 / row["T_K"]
            y = np.log10(
                row["tau_plot_ns"]
            )

            ax.annotate(
                "",
                xy=(x, y - 0.22),
                xytext=(x, y),
                arrowprops=dict(
                    arrowstyle="->",
                    lw=1.0
                )
            )

    # --------------------------------------------------------
    # Slow censored
    # --------------------------------------------------------

    slow = whole[
        whole["status"] == "slow_censored"
    ].copy()

    if len(slow):

        ax.scatter(
            1000.0 / slow["T_K"],
            np.log10(
                slow["tau_plot_ns"]
            ),
            marker="^",
            s=65,
            label=r"$\tau_\alpha >$ maximum analyzed lag"
        )

        for _, row in slow.iterrows():

            x = 1000.0 / row["T_K"]
            y = np.log10(
                row["tau_plot_ns"]
            )

            ax.annotate(
                "",
                xy=(x, y + 0.22),
                xytext=(x, y),
                arrowprops=dict(
                    arrowstyle="->",
                    lw=1.0
                )
            )

    ax.set_xlabel(
        r"$1000/T$ (K$^{-1}$)"
    )

    ax.set_ylabel(
        r"$\log_{10}\tau_\alpha$ (ns)"
    )

    ax.set_title(
        "Resolved and Censored Structural Relaxation"
    )

    ax.grid(
        alpha=0.2
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    fig.savefig(
        out_prefix
        + "_01_censored_relaxation.png",
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        out_prefix
        + "_01_censored_relaxation.pdf",
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# PLOT 2
#
# REPLICA SCATTER
#
# Purpose:
# show reproducibility / divergence among independent replicas
#
# Only ACTUAL resolved tau values are connected.
# Censored observations are shown separately.
# ============================================================

def plot_replica_scatter(df, out_prefix):

    """Replica relaxation times at each temperature.

    Parameters
    ----------
    df : 
        Input table.
    out_prefix : 
        Output path without a file extension.

    Returns
    -------
    result
        Replica relaxation times at each temperature.
    """
    whole = df[
        df["block"] == "whole"
    ].copy()

    fig, ax = plt.subplots(
        figsize=(7.6, 5.6)
    )

    replicas = sorted(
        whole["replica"]
        .dropna()
        .unique()
    )

    for r in replicas:

        d = whole[
            whole["replica"] == r
        ].copy()

        if "status" in d.columns:

            resolved = d[
                d["status"] == "resolved"
            ].copy()

        else:

            resolved = d[
                np.isfinite(
                    d["tau_1e_ns"]
                )
            ].copy()

        resolved = resolved.sort_values(
            "T_K",
            ascending=False
        )

        if len(resolved):

            ax.plot(
                resolved["T_K"],
                np.log10(
                    resolved["tau_1e_ns"]
                ),
                marker="o",
                linewidth=1.3,
                markersize=6,
                label=f"Replica {r}"
            )

        # Censored points kept visible,
        # but never connected as measured tau values.

        if (
            "status" in d.columns
            and
            "tau_plot_ns" in d.columns
        ):

            fast = d[
                d["status"] ==
                "fast_censored"
            ]

            slow = d[
                d["status"] ==
                "slow_censored"
            ]

            if len(fast):

                ax.scatter(
                    fast["T_K"],
                    np.log10(
                        fast["tau_plot_ns"]
                    ),
                    marker="v",
                    s=48
                )

            if len(slow):

                ax.scatter(
                    slow["T_K"],
                    np.log10(
                        slow["tau_plot_ns"]
                    ),
                    marker="^",
                    s=48
                )

    ax.invert_xaxis()

    ax.set_xlabel(
        "Temperature (K)"
    )

    ax.set_ylabel(
        r"$\log_{10}\tau_\alpha$ (ns)"
    )

    ax.set_title(
        "Replica Reproducibility of Structural Relaxation"
    )

    ax.grid(
        alpha=0.2
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    fig.savefig(
        out_prefix
        + "_02_replica_scatter.png",
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        out_prefix
        + "_02_replica_scatter.pdf",
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# CALCULATE BLOCK CV
#
# CV_tau =
#
# SD(block tau_alpha) / mean(block tau_alpha)
#
# Only resolved block tau values are used.
# ============================================================

def calculate_block_cv(df):

    """Coefficient of variation of the block relaxation times.

    Parameters
    ----------
    df : 
        Input table.

    Returns
    -------
    result
        Coefficient of variation of the block relaxation times.
    """
    blocks = df[
        df["block"].isin(
            [
                "block1",
                "block2",
                "block3"
            ]
        )
    ].copy()

    rows = []

    for (
        T,
        replica
    ), d in blocks.groupby(
        [
            "T_K",
            "replica"
        ]
    ):

        # If censoring status exists,
        # only true resolved measurements count.

        if "status" in d.columns:

            d = d[
                d["status"] == "resolved"
            ]

        vals = (
            d["tau_1e_ns"]
            .to_numpy(dtype=float)
        )

        vals = vals[
            np.isfinite(vals)
            &
            (vals > 0)
        ]

        n = len(vals)

        if n >= 2:

            mean_tau = np.mean(vals)

            cv = (
                np.std(
                    vals,
                    ddof=1
                )
                /
                mean_tau
            )

        else:

            mean_tau = np.nan
            cv = np.nan

        rows.append({

            "T_K":
                T,

            "replica":
                replica,

            "n_resolved_blocks":
                n,

            "block_tau_mean_ns":
                mean_tau,

            "block_tau_cv":
                cv,

            "stationary_cv_lt_0p25":
                bool(
                    np.isfinite(cv)
                    and
                    cv < 0.25
                )
        })

    return pd.DataFrame(rows)


# ============================================================
# PLOT 3
#
# BLOCK-STATIONARITY METRIC
# ============================================================

def plot_stationarity(
    stationarity,
    out_prefix
):

    """Block relaxation times used to look for drift inside a trajectory.

    Parameters
    ----------
    stationarity : 
        Block-resolved relaxation times used to test drift.
    out_prefix : 
        Output path without a file extension.

    Returns
    -------
    result
        Block relaxation times used to look for drift inside a trajectory.
    """
    fig, ax = plt.subplots(
        figsize=(7.6, 5.6)
    )

    for r in sorted(
        stationarity[
            "replica"
        ].unique()
    ):

        d = stationarity[
            stationarity[
                "replica"
            ] == r
        ].copy()

        d = d.sort_values(
            "T_K",
            ascending=False
        )

        valid = d[
            np.isfinite(
                d["block_tau_cv"]
            )
        ]

        if len(valid):

            ax.plot(
                valid["T_K"],
                valid["block_tau_cv"],
                marker="o",
                linewidth=1.4,
                markersize=6,
                label=f"Replica {r}"
            )

    # Original Stage-3 stationarity criterion

    ax.axhline(
        0.25,
        linestyle="--",
        linewidth=1.50,
        label=r"Stationarity threshold: $CV_\tau=0.250$"
    )

    ax.invert_xaxis()

    ax.set_xlabel(
        "Temperature (K)"
    )

    ax.set_ylabel(
        r"Block relaxation-time CV, $CV_\tau$"
    )

    ax.set_title(
        "Block-to-Block Stationarity of "
        r"$\tau_\alpha$"
    )

    ax.grid(
        alpha=0.2
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    fig.savefig(
        out_prefix
        + "_03_block_stationarity.png",
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        out_prefix
        + "_03_block_stationarity.pdf",
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    """Write the relaxation-time figures.

    Returns
    -------
    result
        Write the relaxation-time figures.
    """
    ap = argparse.ArgumentParser(
        description=(
            "Stage 4 visualization: censoring, replica "
            "reproducibility, and block stationarity."
        )
    )

    ap.add_argument(
        "-i",
        "--input",
        required=True,
        help=(
            "Full Stage-4 master CSV containing "
            "whole + block rows"
        )
    )

    ap.add_argument(
        "--out-prefix",
        default="./outputs_merged/stage4"
    )

    args = ap.parse_args()

    df = load_data(
        args.input
    )

    # --------------------------------------------------------
    # Figures 1 and 2
    # --------------------------------------------------------

    plot_censored(
        df,
        args.out_prefix
    )

    plot_replica_scatter(
        df,
        args.out_prefix
    )

    # --------------------------------------------------------
    # Stationarity
    # --------------------------------------------------------

    stationarity = calculate_block_cv(
        df
    )

    stationarity_file = (
        args.out_prefix
        + "_stationarity.csv"
    )

    stationarity.to_csv(
        stationarity_file,
        index=False
    )

    plot_stationarity(
        stationarity,
        args.out_prefix
    )

    # --------------------------------------------------------
    # Print useful table
    # --------------------------------------------------------

    print()
    print(
        "=============================================="
    )
    print(
        "STAGE 4 BLOCK STATIONARITY"
    )
    print(
        "=============================================="
    )
    print()

    print(
        stationarity
        .sort_values(
            [
                "T_K",
                "replica"
            ],
            ascending=[
                False,
                True
            ]
        )
        .to_string(
            index=False
        )
    )

    print()
    print("Wrote:")
    print(
        args.out_prefix
        + "_01_censored_relaxation.png"
    )
    print(
        args.out_prefix
        + "_02_replica_scatter.png"
    )
    print(
        args.out_prefix
        + "_03_block_stationarity.png"
    )
    print(
        stationarity_file
    )


if __name__ == "__main__":
    main()
