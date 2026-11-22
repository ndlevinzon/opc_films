#!/usr/bin/env python3

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.rates import parse_rate
from opcfilm.config import CONFIG

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from opcfilm.plotting.conventions import install_figure_conventions
install_figure_conventions()  # temperature axes hot to cold; legends outside panels

from scipy.optimize import brentq


R_GAS = CONFIG.arrhenius.gas_constant_J_mol_K


# ============================================================
# Arrhenius model
#
# ln(tau) = ln(tau0) + B/T
#
# B = Ea/R
# ============================================================

def arrhenius_ln_tau(T, ln_tau0, B):
    """Natural log of an Arrhenius relaxation time.

    Parameters
    ----------
    T : 
        Temperature in kelvin.
    ln_tau0 : 
        Natural log of the Arrhenius prefactor.
    B : 
        Arrhenius slope B = Ea/R, in kelvin.

    Returns
    -------
    result
        Natural log of an Arrhenius relaxation time.
    """
    return ln_tau0 + B / T


def fit_arrhenius(T, tau):

    """Linear fit of ln(tau) against 1/T.

    Parameters
    ----------
    T : 
        Temperature in kelvin.
    tau : 
        Relaxation time.

    Returns
    -------
    result
        Linear fit of ln(tau) against 1/T.
    """
    x = 1.0 / np.asarray(T, dtype=float)
    y = np.log(np.asarray(tau, dtype=float))

    B, ln_tau0 = np.polyfit(
        x,
        y,
        1
    )

    yfit = ln_tau0 + B * x

    ss_res = np.sum(
        (y - yfit) ** 2
    )

    ss_tot = np.sum(
        (y - np.mean(y)) ** 2
    )

    if ss_tot > 0:
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = np.nan

    tau0 = math.exp(ln_tau0)

    Ea_kJmol = (
        B * R_GAS / 1000.0
    )

    return {
        "ln_tau0": ln_tau0,
        "tau0_ns": tau0,
        "B_K": B,
        "Ea_kJmol": Ea_kJmol,
        "R2": r2
    }


# ============================================================
# Cooling-rate Tg criterion
#
# tau(Tg) =
#
# 1 / [ Rcool * |d ln(tau)/dT| ]
#
# For Arrhenius:
#
# ln tau = ln tau0 + B/T
#
# |d ln tau/dT| = B/T^2
#
# therefore:
#
# tau(Tg) = Tg^2 / (Rcool * B)
# ============================================================

def arrhenius_tau(T, fit):

    """Arrhenius relaxation time at one or more temperatures.

    Parameters
    ----------
    T : 
        Temperature in kelvin.
    fit : 
        Arrhenius or energy-fit dictionary.

    Returns
    -------
    result
        Arrhenius relaxation time at one or more temperatures.
    """
    return np.exp(
        fit["ln_tau0"]
        +
        fit["B_K"] / T
    )


def kinetic_timescale(T, cooling_rate, B):

    """Cooling time Tg^2 / (Rc B) from Eq. 3.

    Parameters
    ----------
    T : 
        Temperature in kelvin.
    cooling_rate : 
        Cooling rate in K/ns.
    B : 
        Arrhenius slope B = Ea/R, in kelvin.

    Returns
    -------
    result
        Cooling time Tg^2 / (Rc B) from Eq. 3.
    """
    return (
        T**2
        /
        (cooling_rate * B)
    )


def solve_tg_arrhenius(
    fit,
    cooling_rate,
    Tmin=100.0,
    Tmax=300
):

    """Temperature at which the Arrhenius time equals the cooling time.

    Parameters
    ----------
    fit : 
        Arrhenius or energy-fit dictionary.
    cooling_rate : 
        Cooling rate in K/ns.
    Tmin : 
        Lower temperature bound in kelvin.
    Tmax : 
        Upper temperature bound in kelvin.

    Returns
    -------
    result
        Temperature at which the Arrhenius time equals the cooling time.
    """
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
        tau = arrhenius_tau(
            T,
            fit
        )

        criterion = kinetic_timescale(
            T,
            cooling_rate,
            fit["B_K"]
        )

        return tau - criterion

    return brentq(
        f,
        Tmin,
        Tmax
    )


# ============================================================
# Build temperature-level summary
# ============================================================

def spacing_by_temperature(df):
    """One saved-frame spacing for each temperature.

    Parameters
    ----------
    df : pandas.DataFrame
        Tau table with ``T_K`` and ``frame_ps_selected``.

    Returns
    -------
    dict
        Temperature in kelvin to spacing in picoseconds. Temperatures may differ.

    Raises
    ------
    RuntimeError
        If the column is missing, or one temperature has more than one spacing.
    """
    from opcfilm.thermo.arrest import spacing_by_temperature as shared

    return shared(df)


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
    """
    from opcfilm.thermo.arrest import spacing_at as shared

    return shared(spacings, temperature)


def summarize_temperatures(df):

    """Geometric-mean relaxation time of the resolved replicas.

    Parameters
    ----------
    df : 
        Input table.

    Returns
    -------
    result
        Geometric-mean relaxation time of the resolved replicas.
    """
    rows = []

    for T in sorted(
        df["T_K"].unique(),
        reverse=True
    ):

        d = df[
            df["T_K"] == T
        ]

        resolved = d[
            d["status"] == "resolved"
        ]

        fast = d[
            d["status"] == "fast_censored"
        ]

        slow = d[
            d["status"] == "slow_censored"
        ]

        row = {
            "T_K": T,
            "n_total": len(d),
            "n_resolved": len(resolved),
            "n_fast_censored": len(fast),
            "n_slow_censored": len(slow),
            "tau_geom_mean_ns": np.nan,
            "tau_arith_mean_ns": np.nan,
            "tau_sd_ns": np.nan,
            "log10_tau_mean": np.nan,
            "log10_tau_sd": np.nan
        }

        if len(resolved) > 0:

            tau = resolved[
                "tau_1e_ns"
            ].to_numpy(dtype=float)

            logtau = np.log10(tau)

            row[
                "tau_geom_mean_ns"
            ] = np.exp(
                np.mean(np.log(tau))
            )

            row[
                "tau_arith_mean_ns"
            ] = np.mean(tau)

            if len(tau) >= 2:

                row[
                    "tau_sd_ns"
                ] = np.std(
                    tau,
                    ddof=1
                )

                row[
                    "log10_tau_sd"
                ] = np.std(
                    logtau,
                    ddof=1
                )

            row[
                "log10_tau_mean"
            ] = np.mean(logtau)

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main():

    """Fit the dynamic glass temperature at one cooling rate.

    Returns
    -------
    result
        Fit the dynamic glass temperature at one cooling rate.
    """
    ap = argparse.ArgumentParser(
        description=(
            "Stage 5: fit equilibrium alpha-relaxation "
            "times and estimate kinetic Tg."
        )
    )

    ap.add_argument(
        "-i",
        "--input",
        required=True,
        help="Stage-4 whole-trajectory CSV"
    )

    ap.add_argument(
        "--fit-tmin",
        type=float,
        default=CONFIG.arrhenius.window_K[0],
        help="Lowest T used for equilibrium fit"
    )

    ap.add_argument(
        "--fit-tmax",
        type=float,
        default=CONFIG.arrhenius.window_K[1],
        help="Highest T used for equilibrium fit"
    )

    ap.add_argument(
        "--cooling-rate",
        type=float,
        default=4.44,
        help="Physical cooling rate in K/ns. A directory code such as 4 is rejected."
    )

    ap.add_argument(
        "--out-prefix",
        required=True
    )
    ap.add_argument(
        "--frame-ps",
        type=float,
        default=None,
        help="Optional. Accepted only when every temperature has this spacing.",
    )
    ap.add_argument(
        "--tg-uncertainty-K",
        type=float,
        default=None,
        help="If given, a half-frame shift of tau(230 K) must move Tg by less than this.",
    )

    args = ap.parse_args()
    args.cooling_rate = parse_rate(args.cooling_rate)


    # ========================================================
    # Read data
    # ========================================================

    df = pd.read_csv(
        args.input
    )


    required = [
        "T_K",
        "replica",
        "tau_1e_ns",
        "status"
    ]

    missing = [
        x for x in required
        if x not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing required columns: "
            + ", ".join(missing)
        )
    spacings = spacing_by_temperature(df)
    if args.frame_ps is not None:
        mismatched = [
            temperature for temperature, spacing in spacings.items()
            if abs(spacing - float(args.frame_ps)) > 1e-3
        ]
        if mismatched:
            raise RuntimeError(
                f"--frame-ps {args.frame_ps:g} is one spacing. "
                f"These temperatures differ: {mismatched}."
            )


    # ========================================================
    # Temperature summaries
    # ========================================================

    summary = summarize_temperatures(
        df
    )
    summary["frame_ps_selected"] = [spacing_at(spacings, temperature) for temperature in summary["T_K"]]

    summary_file = (
        args.out_prefix
        + "_temperature_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )


    # ========================================================
    # Select equilibrium branch
    #
    # Require:
    #   all 3 replicas resolved
    #   T within specified fit range
    # ========================================================

    eq = summary[
        (summary["T_K"] >= args.fit_tmin)
        &
        (summary["T_K"] <= args.fit_tmax)
        &
        (summary["n_resolved"] == 3)
    ].copy()


    if len(eq) < 3:

        raise RuntimeError(
            "Fewer than 3 fully resolved equilibrium "
            "temperatures available."
        )


    Tfit = eq[
        "T_K"
    ].to_numpy(dtype=float)

    tau_fit = eq[
        "tau_geom_mean_ns"
    ].to_numpy(dtype=float)


    # ========================================================
    # Arrhenius fit
    # ========================================================

    fit = fit_arrhenius(
        Tfit,
        tau_fit
    )


    # ========================================================
    # Kinetic Tg
    # ========================================================

    Tg = solve_tg_arrhenius(
        fit,
        args.cooling_rate,
        Tmin=150,
        Tmax=230
    )


    tau_Tg = arrhenius_tau(
        Tg,
        fit
    )

    criterion_Tg = kinetic_timescale(
        Tg,
        args.cooling_rate,
        fit["B_K"]
    )


    # ========================================================
    # Fit summary
    # ========================================================

    fit_summary = pd.DataFrame(
        [{
            "model": "Arrhenius",
            "fit_Tmin_K": args.fit_tmin,
            "fit_Tmax_K": args.fit_tmax,
            "n_temperatures": len(eq),
            "tau0_ns": fit["tau0_ns"],
            "B_K": fit["B_K"],
            "Ea_kJmol": fit["Ea_kJmol"],
            "R2_log_space": fit["R2"],
            "cooling_rate_K_per_ns":
                args.cooling_rate,
            "Tg_K":
                Tg,
            "tau_at_Tg_ns":
                tau_Tg,
            "kinetic_timescale_at_Tg_ns":
                criterion_Tg
        }]
    )

    fit_file = (
        args.out_prefix
        + "_fit_summary.csv"
    )

    fit_summary.to_csv(
        fit_file,
        index=False
    )

    window_rows = []
    resolved = summary[summary["n_resolved"] == 3]
    for lower in (CONFIG.arrhenius.window_K[0], *CONFIG.arrhenius.sensitivity_lower_K):
        window = resolved[
            (resolved["T_K"] >= lower) & (resolved["T_K"] <= args.fit_tmax)
        ]
        if len(window) < 3:
            continue
        window_fit = fit_arrhenius(window["T_K"], window["tau_geom_mean_ns"])
        window_tg = solve_tg_arrhenius(
            window_fit, args.cooling_rate, Tmin=150, Tmax=230
        )
        window_rows.append({
            "fit_Tmin_K": lower,
            "fit_Tmax_K": args.fit_tmax,
            "n_temperatures": len(window),
            "Ea_kJmol": window_fit["Ea_kJmol"],
            "Tg_K": window_tg,
            "diagnostic_only": lower < 210.0,
        })
    pd.DataFrame(window_rows).to_csv(
        args.out_prefix + "_arrhenius_window_sensitivity.csv", index=False
    )

    anchor = int(np.argmin(np.abs(Tfit - 230.0)))
    anchor_frame_ps = spacing_at(spacings, Tfit[anchor])
    half_ns = 0.5 * anchor_frame_ps * 1.0e-3
    shift_rows = []
    for sign in (-1.0, 1.0):
        shifted = tau_fit.copy()
        shifted[anchor] = shifted[anchor] + sign * half_ns
        if np.any(shifted <= 0):
            continue
        shifted_fit = fit_arrhenius(Tfit, shifted)
        shifted_tg = solve_tg_arrhenius(
            shifted_fit, args.cooling_rate, Tmin=150, Tmax=230
        )
        shift_rows.append({
            "anchor_T_K": float(Tfit[anchor]),
            "frame_ps_selected": anchor_frame_ps,
            "tau_shift_ns": sign * half_ns,
            "Tg_K": shifted_tg,
            "delta_Tg_K": shifted_tg - Tg,
        })
    shift_table = pd.DataFrame(shift_rows)
    shift_table.to_csv(args.out_prefix + "_tau230_half_frame.csv", index=False)
    if len(shift_table) and args.tg_uncertainty_K is not None:
        largest = float(np.nanmax(np.abs(shift_table["delta_Tg_K"])))
        if largest >= args.tg_uncertainty_K:
            raise RuntimeError(
                f"Shifting tau({Tfit[anchor]:.0f} K) by half a frame moves Tg by "
                f"{largest:.3f} K, which is not below the {args.tg_uncertainty_K:g} K uncertainty."
            )


    # ========================================================
    # Print results
    # ========================================================

    print()
    print(
        "=============================================="
    )
    print(
        "EQUILIBRIUM RELAXATION DATA"
    )
    print(
        "=============================================="
    )

    print(
        eq[
            [
                "T_K",
                "tau_geom_mean_ns",
                "tau_sd_ns",
                "log10_tau_sd"
            ]
        ].to_string(index=False)
    )


    print()
    print(
        "=============================================="
    )
    print(
        "ARRHENIUS FIT"
    )
    print(
        "=============================================="
    )

    print(
        f"T range: "
        f"{args.fit_tmin:.1f}"
        f" - "
        f"{args.fit_tmax:.1f} K"
    )

    print(
        f"B = "
        f"{fit['B_K']:.3f} K"
    )

    print(
        f"Ea = "
        f"{fit['Ea_kJmol']:.3f} kJ/mol"
    )

    print(
        f"tau0 = "
        f"{fit['tau0_ns']:.6e} ns"
    )

    print(
        f"R^2 = "
        f"{fit['R2']:.6f}"
    )


    print()
    print(
        "=============================================="
    )
    print(
        "KINETIC GLASS TRANSITION"
    )
    print(
        "=============================================="
    )

    print(
        f"Cooling rate = "
        f"{args.cooling_rate:.3f} K/ns"
    )

    print(
        f"Tg = "
        f"{Tg:.3f} K"
    )

    print(
        f"tau(Tg) = "
        f"{tau_Tg:.3f} ns"
    )

    print(
        f"cooling timescale = "
        f"{criterion_Tg:.3f} ns"
    )


    # ========================================================
    # Plot
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(8.0, 6.0)
    )


    # --------------------------------------------------------
    # Individual replica data
    # --------------------------------------------------------

    resolved = df[
        df["status"] == "resolved"
    ]

    for r in sorted(
        resolved["replica"].unique()
    ):

        d = resolved[
            resolved["replica"] == r
        ]

        ax.scatter(
            1000.0 / d["T_K"],
            np.log10(
                d["tau_1e_ns"]
            ),
            s=45,
            alpha=0.65,
            label=f"Replica {r}"
        )


    # --------------------------------------------------------
    # Equilibrium geometric means
    # --------------------------------------------------------

    ax.errorbar(
        1000.0 / eq["T_K"],
        np.log10(
            eq["tau_geom_mean_ns"]
        ),
        yerr=eq[
            "log10_tau_sd"
        ],
        fmt="o",
        markersize=8,
        capsize=4,
        linewidth=1.50,
        label="Equilibrium replica mean"
    )


    # --------------------------------------------------------
    # Arrhenius fit
    # --------------------------------------------------------

    TT = np.linspace(
        Tg - 5,
        args.fit_tmax + 5,
        400
    )

    tau_line = arrhenius_tau(
        TT,
        fit
    )

    ax.plot(
        1000.0 / TT,
        np.log10(
            tau_line
        ),
        linewidth=2,
        label="Arrhenius fit"
    )


    # --------------------------------------------------------
    # Mark Tg
    # --------------------------------------------------------

    ax.axvline(
        1000.0 / Tg,
        linestyle="--",
        linewidth=1.50
    )

    ax.text(
        1000.0 / Tg,
        np.log10(tau_Tg),
        f"  Tg = {Tg:.1f} K",
        va="bottom"
    )


    ax.set_xlabel(
        r"$1000/T$ (K$^{-1}$)"
    )

    ax.set_ylabel(
        r"$\log_{10}\tau_\alpha$ (ns)"
    )

    ax.set_title(
        "OPC Water-Slab Structural Relaxation"
    )

    ax.grid(
        alpha=0.2
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()


    png = (
        args.out_prefix
        + "_arrhenius.png"
    )

    pdf = (
        args.out_prefix
        + "_arrhenius.pdf"
    )

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf,
        bbox_inches="tight"
    )

    plt.close(fig)


    print()
    print(
        f"Wrote {summary_file}"
    )

    print(
        f"Wrote {fit_file}"
    )

    print(
        f"Wrote {png}"
    )

    print(
        f"Wrote {pdf}"
    )


if __name__ == "__main__":
    main()
