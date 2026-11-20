"""Figures for depth-resolved local order, structure factors, and Bragg checks.

``--data-dir`` is the directory that contains the CSV and NPZ files written
by ``local_order.py``. Pass ``--condition cool``, ``warm``, or ``both``.
Figures are written to ``--out`` and are not shown on screen.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.stats import gaussian_kde
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import binned_statistic_2d
from mpl_toolkits.mplot3d import proj3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.constants import BULK_HALFWIDTH_A

def condition_temp(condition):
    """Temperature label of a Cool or Warm condition.

    Parameters
    ----------
    condition : 
        Liquid or glass label, Cool or Warm.

    Returns
    -------
    result
        Temperature label of a Cool or Warm condition.
    """
    return condition_temperature_labels.get(condition, condition)

def make_plot_title(model, plot_name, condition=None):
    """Figure title from the model, plot name, and condition.

    Parameters
    ----------
    model : 
        Water-model display name.
    plot_name : 
        Short name used in the title and file name.
    condition : 
        Liquid or glass label, Cool or Warm.

    Returns
    -------
    result
        Figure title from the model, plot name, and condition.
    """
    if condition is None:
        return f"{model} {plot_name}"
    return f"{model} {plot_name} @ {condition_temp(condition)}"

def safe_filename(text):
    """File name with spaces and punctuation removed.

    Parameters
    ----------
    text : 
        Text used to build a file name.

    Returns
    -------
    result
        File name with spaces and punctuation removed.
    """
    return (
        text.replace(" ", "_")
        .replace("@", "at")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace("-", "_")
    )

def plot_z_profile_pairwise(column, sd_column, xlabel, filename, title):
    """Order-parameter profile of each loaded condition.

    Parameters
    ----------
    column : 
        Data-column name.
    sd_column : 
        Column containing the standard deviation.
    xlabel : 
        Horizontal axis label.
    filename : 
        Input or output file name.
    title : 
        Figure title.

    Returns
    -------
    result
        Order-parameter profile of each loaded condition.
    """
    fig, ax = plt.subplots(figsize=(7, 8))
    for model, cond_dict in profiles.items():
        color = model_colors.get(model, None)
        for condition, df in cond_dict.items():
            if column not in df.columns:
                continue
            z = df["z_center"]
            y = df[column]
            sd = df[sd_column] if sd_column in df.columns else None
            linestyle = conditions[condition]["linestyle"]
            ax.plot(
                y,
                z,
                linewidth=2.5,
                linestyle=linestyle,
                label=f"{model} {condition}",
                color=color,
            )
            if sd is not None:
                ax.fill_betweenx(
                    z,
                    y - sd,
                    y + sd,
                    alpha=0.10,
                    color=color,
                    label="Per-molecule distribution width",
                )
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Z Coordinate ($\mathrm{\AA}$)")
    ax.set_title(title, pad=16)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.grid(axis="both", linestyle="--", alpha=0.35)
    ax.legend(frameon=True, edgecolor="black", loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, filename), dpi=300, bbox_inches="tight")
    plt.close()

def plot_delta_profile(column, xlabel, filename, title):
    """Difference between two conditions of one order parameter.

    Parameters
    ----------
    column : 
        Data-column name.
    xlabel : 
        Horizontal axis label.
    filename : 
        Input or output file name.
    title : 
        Figure title.

    Returns
    -------
    result
        Difference between two conditions of one order parameter.
    """
    fig, ax = plt.subplots(figsize=(7, 8))
    for model, cond_dict in profiles.items():
        if "Warm" not in cond_dict or "Cool" not in cond_dict:
            continue
        warm = cond_dict["Warm"]
        cool = cond_dict["Cool"]
        if column not in warm.columns or column not in cool.columns:
            continue
        merged = warm[["z_center", column]].merge(
            cool[["z_center", column]],
            on="z_center",
            suffixes=("_warm", "_cool"),
        )
        merged["delta"] = merged[f"{column}_cool"] - merged[f"{column}_warm"]
        color = model_colors.get(model, None)
        ax.plot(
            merged["delta"],
            merged["z_center"],
            linewidth=2.5,
            label=f"{model}: Cool - Warm",
            color=color,
        )
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Z Coordinate ($\mathrm{\AA}$)")
    ax.set_title(title, pad=16)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.grid(axis="both", linestyle="--", alpha=0.35)
    ax.legend(frameon=True, edgecolor="black", loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, filename), dpi=300, bbox_inches="tight")
    plt.close()

def plot_distribution_by_z(
    df,
    value_col,
    xlabel,
    filename,
    model,
    condition,
    plot_name,
    xlim=None,
    bins=60,
    use_kde=True,
):
    """Distribution of one order parameter in each z window.

    Parameters
    ----------
    df : 
        Input table.
    value_col : 
        Column of the order parameter being histogrammed.
    xlabel : 
        Horizontal axis label.
    filename : 
        Input or output file name.
    model : 
        Water-model display name.
    condition : 
        Liquid or glass label, Cool or Warm.
    plot_name : 
        Short name used in the title and file name.
    xlim : 
        Horizontal axis limits.
    bins : 
        Histogram bin edges or bin count.
    use_kde : 
        Draw a kernel density estimate instead of a histogram.

    Returns
    -------
    result
        Distribution of one order parameter in each z window.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    z_values = np.array(sorted(df["z_center"].dropna().unique()))
    z_values = z_values[np.argsort(np.abs(z_values))]
    cmap = plt.get_cmap("turbo_r")
    norm = Normalize(vmin=0.0, vmax=np.nanmax(np.abs(z_values)))
    for zc in z_values:
        sub = df[df["z_center"] == zc][value_col].dropna()
        if len(sub) < 20:
            continue
        color = cmap(norm(abs(zc)))
        if use_kde:
            x_min = xlim[0] if xlim is not None else sub.min()
            x_max = xlim[1] if xlim is not None else sub.max()
            x_grid = np.linspace(x_min, x_max, 400)
            kde = gaussian_kde(sub)
            y = kde(x_grid)
            ax.plot(x_grid, y, linewidth=2.0, color=color, alpha=0.85)
        else:
            ax.hist(
                sub,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2.0,
                color=color,
                alpha=0.85,
            )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized Probability Density")
    ax.set_title(make_plot_title(model, plot_name, condition), pad=16)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.grid(axis="both", linestyle="--", alpha=0.35)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$|z|$ from Slab Center ($\mathrm{\AA}$)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, filename), dpi=300, bbox_inches="tight")
    plt.close()

def plot_chillplus_by_distance_from_center(path, display_name, safe_name, condition):
    """CHILL+ class fractions against distance from the slab center.

    Parameters
    ----------
    path : 
        Input file or directory.
    display_name : 
        Label printed on the figure.
    safe_name : 
        File-name-safe model label.
    condition : 
        Liquid or glass label, Cool or Warm.

    Returns
    -------
    result
        CHILL+ class fractions against distance from the slab center.
    """
    df = pd.read_csv(path)
    df["abs_z_center"] = df["z_center"].abs()
    chill_labels = {
        0: "Disordered Water",
        1: "Hexagonal Ice",
        2: "Cubic Ice",
        3: "Interfacial Ice",
        4: "Gas Hydrate",
        5: "Interfacial Gas Hydrate",
    }
    fraction_cols = [
        c for c in df.columns
        if c.startswith("fraction_type_")
    ]
    if len(fraction_cols) == 0:
        print(f"No CHILL+ fraction columns found in {path}")
        print(df.columns.tolist())
        return
    def get_type_id(col):
        """CHILL+ integer stored in a column name.

        Parameters
        ----------
        col : 
            Data-column name.

        Returns
        -------
        result
            CHILL+ integer stored in a column name.
        """
        return int(col.replace("fraction_type_", ""))
    fraction_cols = sorted(fraction_cols, key=get_type_id)
    grouped = (
        df.groupby("abs_z_center")[fraction_cols]
        .mean()
        .reset_index()
        .sort_values("abs_z_center")
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(grouped))
    for col in fraction_cols:
        type_id = get_type_id(col)
        label = chill_labels.get(type_id, f"Type {type_id}")
        ax.bar(
            grouped["abs_z_center"],
            grouped[col],
            bottom=bottom,
            width=8,
            label=label,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.4,
        )
        bottom += grouped[col].to_numpy()
    ax.set_xlabel(r"Distance from Slab Center $|z|$ ($\mathrm{\AA}$)")
    ax.set_ylabel("Fraction of Waters")
    ax.set_title(
        make_plot_title(
            display_name,
            "CHILL+ Water Classification vs Distance from Center",
            condition,
        ),
    )
    ax.set_xlim(0, 70)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    legend = ax.legend(
        title="CHILL+ Class",
        frameon=True,
        edgecolor="black",
        loc="center left",
        bbox_to_anchor=(1.02, 0.62),
    )
    # ------------------------------------------------------------
    # Ice / non-ice summary box
    # ------------------------------------------------------------
    ice_types = [1, 2, 3]  # Hexagonal, cubic, interfacial ice
    nonice_types = [0, 4, 5]  # Disordered water, hydrate classes
    count_cols = [
        c for c in df.columns
        if c.startswith("count_type_")
    ]
    def get_count_col(type_id):
        """Count column that belongs to one CHILL+ class.

        Parameters
        ----------
        type_id : 
            CHILL+ structure-type integer.

        Returns
        -------
        result
            Count column that belongs to one CHILL+ class.
        """
        col = f"count_type_{type_id}"
        return col if col in df.columns else None
    ice_count = 0.0
    for t in ice_types:
        col = get_count_col(t)
        if col is not None:
            ice_count += df[col].sum()
    nonice_count = 0.0
    for t in nonice_types:
        col = get_count_col(t)
        if col is not None:
            nonice_count += df[col].sum()
    total_count = ice_count + nonice_count
    ice_pct = 100.0 * ice_count / total_count if total_count > 0 else np.nan
    nonice_pct = 100.0 * nonice_count / total_count if total_count > 0 else np.nan
    summary_text = (
        rf"Ice waters: {ice_count:.0f} ({ice_pct:.2f}%)"
        "\n"
        rf"Non-ice waters: {nonice_count:.0f} ({nonice_pct:.2f}%)"
    )
    ax.text(
        1.02,
        0.28,
        summary_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="white",
            edgecolor="black",
            alpha=0.950,
        ),
    )
    plt.tight_layout(rect=[0, 0.03, 0.78, 1])
    plt.savefig(
        os.path.join(
            outdir,
            f"{safe_name}_{condition}_CHILLplus_by_abs_z.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

def load_q_sk_lsi_combined(path, condition):
    # ========================================================
    # FILE NAMING
    # ========================================================
    #
    # Cool:
    #   water_q_per_water_distribution.csv
    #   water_Sk_per_water_distribution.csv
    #   water_LSI_per_water_distribution.csv
    #
    # Warm:
    #   warm_water_q_per_water_distribution.csv
    #   warm_water_Sk_per_water_distribution.csv
    #   warm_water_LSI_per_water_distribution.csv
    # ========================================================
    """Per-water q, Sk, and LSI joined on the molecule index.

    Parameters
    ----------
    path : 
        Input file or directory.
    condition : 
        Liquid or glass label, Cool or Warm.

    Returns
    -------
    result
        Per-water q, Sk, and LSI joined on the molecule index.
    """
    if condition == "Cool":
        prefix = ""
    elif condition == "Warm":
        prefix = "warm_"
    else:
        raise ValueError(
            f"Unknown condition: {condition}"
        )
    def _existing(*names):
        for name in names:
            candidate = os.path.join(path, name)
            if os.path.exists(candidate):
                return candidate
        return os.path.join(path, names[0])

    q_file = _existing(
        "q_per_water_distribution.csv",
        f"{prefix}water_q_per_water_distribution.csv",
        f"{prefix}q_per_water_distribution.csv",
    )
    sk_file = _existing(
        "Sk_per_water_distribution.csv",
        f"{prefix}water_Sk_per_water_distribution.csv",
        f"{prefix}Sk_per_water_distribution.csv",
    )
    lsi_file = _existing(
        "LSI_per_water_distribution.csv",
        f"{prefix}water_LSI_per_water_distribution.csv",
        f"{prefix}LSI_per_water_distribution.csv",
    )
    # ========================================================
    # CHECK FILES
    # ========================================================
    files = {
        "q": q_file,
        "Sk": sk_file,
        "LSI": lsi_file,
    }
    missing = [
        f
        for f in files.values()
        if not os.path.exists(f)
    ]
    if missing:
        print(
            f"\nMissing {condition} q/Sk/LSI files:"
        )
        for f in missing:
            print(f"  {f}")
        return None
    print(
        f"\nLoading {condition} local-structure files:"
    )
    for name, f in files.items():
        print(f"  {name}: {f}")
    # ========================================================
    # LOAD
    # ========================================================
    q_df = pd.read_csv(q_file)
    sk_df = pd.read_csv(sk_file)
    lsi_df = pd.read_csv(lsi_file)
    # ========================================================
    # VERIFY NEW FILE FORMAT
    # ========================================================
    required_common = [
        "frame",
        "oxygen_index",
    ]
    for name, data in [
        ("q", q_df),
        ("Sk", sk_df),
        ("LSI", lsi_df),
    ]:
        missing_cols = [
            col
            for col in required_common
            if col not in data.columns
        ]
        if missing_cols:
            raise RuntimeError(
                f"{name} file is missing columns "
                f"{missing_cols}.\n"
                "These appear to be old per-water files. "
                "Rerun the local-structure analysis so that "
                "oxygen_index is written."
            )
    if "z_coord" not in q_df.columns:
        raise RuntimeError(
            "q file does not contain z_coord. "
            "Rerun the local-structure extraction using "
            "the updated script."
        )
    # ========================================================
    # MERGE THE SAME WATER ACROSS ALL THREE METRICS
    # ========================================================
    merge_cols = [
        "frame",
        "oxygen_index",
    ]
    df = (
        q_df[
            merge_cols
            + [
                "z_coord",
                "q",
            ]
        ]
        .merge(
            sk_df[
                merge_cols
                + ["Sk"]
            ],
            on=merge_cols,
            how="inner",
            validate="one_to_one",
        )
        .merge(
            lsi_df[
                merge_cols
                + ["LSI"]
            ],
            on=merge_cols,
            how="inner",
            validate="one_to_one",
        )
    )
    # ========================================================
    # CLEAN
    # ========================================================
    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )
    df = df.dropna(
        subset=[
            "q",
            "Sk",
            "LSI",
            "z_coord",
        ]
    )
    # ========================================================
    # Bulk window shared with the rest of the analysis.
    # ========================================================
    bulk_cutoff = BULK_HALFWIDTH_A
    df = df[
        np.abs(df["z_coord"])
        <= bulk_cutoff
    ].copy()
    # ========================================================
    # DIAGNOSTICS
    # ========================================================
    print(
        f"  q records:   {len(q_df):,}"
    )
    print(
        f"  Sk records:  {len(sk_df):,}"
    )
    print(
        f"  LSI records: {len(lsi_df):,}"
    )
    print(
        f"  Matched q/Sk/LSI records: "
        f"{len(df):,} "
        f"within |z| <= {bulk_cutoff:.1f} A"
    )
    if len(df) > 0:
        print(
            f"  median q   = "
            f"{df['q'].median():.6f}"
        )
        print(
            f"  median Sk  = "
            f"{df['Sk'].median():.6f}"
        )
        print(
            f"  median LSI = "
            f"{df['LSI'].median():.6f} A^2"
        )
    return df

def plot_q_sk_lsi_median_landscape(condition):
    """Joint q-Sk histogram colored by the median LSI.

    Parameters
    ----------
    condition : 
        Liquid or glass label, Cool or Warm.

    Returns
    -------
    result
        Joint q-Sk histogram colored by the median LSI.
    """
    for model_key, model_info in models.items():
        display = model_info["display"]
        safe = model_info["safe"]
        path = model_info["path"]
        df = load_q_sk_lsi_combined(
            path,
            condition
        )
        if df is None or len(df) == 0:
            print(
                f"Missing q/Sk/LSI combined data "
                f"for {display} {condition}"
            )
            continue
        # ========================================================
        # DATA
        # ========================================================
        q = df["q"].to_numpy()
        sk = df["Sk"].to_numpy()
        lsi = df["LSI"].to_numpy()
        # ========================================================
        # BASIC STATISTICS
        # ========================================================
        print("\n" + "=" * 70)
        print(
            f"{display} {condition}: "
            f"bulk |z| <= {BULK_HALFWIDTH_A:g} A"
        )
        print("=" * 70)
        print(
            f"N waters = {len(df):,}"
        )
        print(
            f"q:   median={np.median(q):.5f}, "
            f"5-95%="
            f"{np.percentile(q,5):.5f}-"
            f"{np.percentile(q,95):.5f}"
        )
        print(
            f"Sk:  median={np.median(sk):.7f}, "
            f"5-95%="
            f"{np.percentile(sk,5):.7f}-"
            f"{np.percentile(sk,95):.7f}"
        )
        print(
            f"LSI: median={np.median(lsi):.5f} A^2, "
            f"5-95%="
            f"{np.percentile(lsi,5):.5f}-"
            f"{np.percentile(lsi,95):.5f}"
        )
        # ========================================================
        # PLOT RANGE
        # ========================================================
        q_min = 0.45
        q_max = 1.00
        sk_min = 0.9965
        sk_max = 1.0000
        sk_max = 1.0000
        inside = np.isfinite(q) & np.isfinite(sk) & (q >= q_min) & (sk >= sk_min) & (sk <= sk_max)
        n_finite = int((np.isfinite(q) & np.isfinite(sk)).sum())
        excluded = 1.0 - float(inside.sum()) / n_finite if n_finite else float("nan")
        print(
            f"Joint window q>={q_min}, Sk>={sk_min}: "
            f"{int(inside.sum())} of {n_finite} molecules inside, excluded {excluded:.3%}"
        )
        coverage = os.path.join(outdir, "joint_window_exclusion.csv")
        row = pd.DataFrame([{
            "condition": condition,
            "temperature_K": 78.0 if condition == "Cool" else 300.0,
            "n_finite": n_finite,
            "n_inside_window": int(inside.sum()),
            "fraction_excluded": excluded,
        }])
        row.to_csv(coverage, mode="a", header=not os.path.exists(coverage), index=False)
        # ========================================================
        # BINS
        # ========================================================
        #
        # Fewer bins than the 2D heatmap:
        # thousands of 3D bars become visually cluttered.
        # ========================================================
        N_Q_BINS = 40
        N_SK_BINS = 40
        q_bins = np.linspace(q_min, q_max, N_Q_BINS + 1)
        sk_bins = np.linspace(sk_min, sk_max, N_SK_BINS + 1)
        # ========================================================
        # BIN STATISTICS
        # ========================================================
        median_lsi, q_edges, sk_edges, _ = (
            binned_statistic_2d(
                q,
                sk,
                lsi,
                statistic="median",
                bins=[
                    q_bins,
                    sk_bins,
                ],
            )
        )
        counts, _, _, _ = (
            binned_statistic_2d(
                q,
                sk,
                lsi,
                statistic="count",
                bins=[
                    q_bins,
                    sk_bins,
                ],
            )
        )
        # ========================================================
        # MINIMUM OCCUPANCY
        # ========================================================
        MIN_COUNT = 20
        valid = (
            (counts >= MIN_COUNT)
            &
            np.isfinite(median_lsi)
        )
        print(
            f"Occupied bins >= {MIN_COUNT} waters: "
            f"{np.sum(valid)}"
        )
        # ========================================================
        # BAR COORDINATES
        # ========================================================
        q_left = q_edges[:-1]
        sk_left = sk_edges[:-1]
        dq = np.diff(q_edges)
        dsk = np.diff(sk_edges)
        # Build 2D grids matching counts / median_lsi
        Q, SK = np.meshgrid(
            q_left,
            sk_left,
            indexing="ij"
        )
        DQ, DSK = np.meshgrid(
            dq,
            dsk,
            indexing="ij"
        )
        # Flatten only valid bins
        x = Q[valid]
        y = SK[valid]
        dx = DQ[valid]
        dy = DSK[valid]
        z0 = np.zeros_like(x)
        total_counts = np.nansum(counts)
        heights = counts[valid] / total_counts
        lsi_colors = median_lsi[valid]
        # ========================================================
        # COLOR NORMALIZATION
        # ========================================================
        #
        # Keep this fixed across Warm / Cool comparisons.
        # ========================================================
        vmin = 0.00
        vmax = 0.20
        norm = Normalize(
            vmin=vmin,
            vmax=vmax
        )
        cmap = plt.get_cmap(
            "viridis"
        )
        bar_colors = cmap(
            norm(lsi_colors)
        )
        # ========================================================
        # FIGURE
        # ========================================================
        fig = plt.figure(
            figsize=(12.5, 9.5)
        )
        # IMPORTANT:
        # Disable Matplotlib's automatic collection z-order.
        # We will explicitly control the drawing order of each bar.
        ax = fig.add_subplot(
            111,
            projection="3d",
            computed_zorder=False,
        )
        # ========================================================
        # CAMERA AND LIMITS
        # ========================================================
        ax.set_xlim(
            q_min,
            q_max
        )
        ax.set_ylim(
            sk_min,
            sk_max
        )
        # Avoid overlap between q=1.0 and the minimum Sk tick label.
        # Keep the actual Sk axis/bins starting at 0.9965, but do not
        # display the 0.9965 tick.
        ax.set_yticks(
            np.arange(
                0.9970,
                1.0001,
                0.0005,
            )
        )
        # ========================================================
        # FIXED POPULATION-FRACTION SCALE
        # ========================================================
        POPULATION_ZMAX = 0.040
        ax.set_zlim(
            0.0,
            POPULATION_ZMAX,
        )
        # ax.set_zticks(
        #     np.arange(
        #         0.0,
        #         POPULATION_ZMAX + 0.0001,
        #         0.010,
        #     )
        # )
        # Choose the final camera angle BEFORE calculating depth.
        VIEW_ELEV = 28
        VIEW_AZIM = -60
        ax.view_init(
            elev=VIEW_ELEV,
            azim=VIEW_AZIM,
        )
        # ========================================================
        # BAR WIDTHS
        # ========================================================
        bar_dx = (
            dx * 0.88
        )
        bar_dy = (
            dy * 0.88
        )
        # ========================================================
        # DETERMINE CAMERA DEPTH OF EACH BAR
        # ========================================================
        #
        # We deliberately calculate depth using the CENTER OF THE
        # BAR FOOTPRINT at z=0.
        #
        # This means "front vs back" is determined by where the
        # bar lies in the q-Sk plane, NOT by how tall the bar is.
        #
        # That gives exactly the desired behavior:
        #
        #     rear bars drawn first
        #     front bars drawn last
        #
        # ========================================================
        x_center = (
            x
            + 0.5 * bar_dx
        )
        y_center = (
            y
            + 0.5 * bar_dy
        )
        z_reference = np.zeros_like(
            x_center
        )
        # Projection matrix for the selected camera
        projection_matrix = (
            ax.get_proj()
        )
        _, _, projected_depth = (
            proj3d.proj_transform(
                x_center,
                y_center,
                z_reference,
                projection_matrix,
            )
        )
        # In Matplotlib's projected coordinates:
        #
        # more-negative depth = closer to viewer
        #
        # Therefore:
        #
        # highest / least-negative depth = farthest
        # lowest / most-negative depth   = nearest
        #
        # Draw farthest -> nearest.
        draw_order = (
            np.argsort(
                projected_depth
            )[::-1]
        )
        # ========================================================
        # DRAW EACH BAR SEPARATELY
        # ========================================================
        #
        # This is the important difference from one ax.bar3d()
        # call containing every bar.
        #
        # Each bar is now its own Poly3DCollection, so we can
        # explicitly assign the z-order.
        # ========================================================
        for draw_rank, i in enumerate(
            draw_order
        ):
            bar = ax.bar3d(
                x[i],
                y[i],
                0.0,
                bar_dx[i],
                bar_dy[i],
                heights[i],
                color=bar_colors[i],
                edgecolor="black",
                linewidth=0.08,
                alpha=1.0,
                shade=True,
                # This only controls faces WITHIN this one bar.
                zsort="average",
            )
            # Bars nearer the viewer receive progressively
            # larger zorder values.
            bar.set_zorder(
                10 + draw_rank
            )
        # ========================================================
        # AXIS LABELS
        # ========================================================
        ax.set_xlabel(
            r"Orientational Tetrahedral Order, $q$",
            fontsize=15,
            labelpad=18,
        )
        ax.set_ylabel(
            r"Translational Tetrahedral Order, $S_k$",
            fontsize=15,
            labelpad=22,
        )
        ax.set_zlabel(
            "Population Fraction",
            fontsize=15,
            labelpad=16,
        )
        ax.tick_params(
            axis="x",
            pad=6
        )
        ax.tick_params(
            axis="y",
            pad=8
        )
        ax.tick_params(
            axis="z",
            pad=6
        )
        # ========================================================
        # GRID
        # ========================================================
        ax.grid(
            True,
            alpha=0.20
        )
        # ========================================================
        # TITLE
        # ========================================================
        ax.set_title(
            rf"{display} Local-Structure Population Landscape at "
            rf"{condition_temp(condition)}"
            "\n"
            rf"Central Slab Region ($|z|\leq5$ Å)",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        # ========================================================
        # COLOR BAR = MEDIAN LSI
        # ========================================================
        sm = ScalarMappable(
            norm=norm,
            cmap=cmap
        )
        sm.set_array([])
        cbar = fig.colorbar(
            sm,
            ax=ax,
            pad=0.10,
            fraction=0.035,
            shrink=0.75,
        )
        cbar.set_label(
            r"Median LSI ($\mathrm{\AA}^2$)",
            fontsize=13,
        )
        # # ========================================================
        # # MEDIAN STRUCTURE MARKER + LABEL
        # # ========================================================
        # q_med = np.median(q)
        # sk_med = np.median(sk)
        # iq = np.digitize(q_med, q_edges) - 1
        # isk = np.digitize(sk_med, sk_edges) - 1
        # if (
        #     0 <= iq < counts.shape[0]
        #     and
        #     0 <= isk < counts.shape[1]
        # ):
        #     median_height = counts[iq, isk] / total_counts
        #     marker_z = median_height * 1.08 + 0.002
        #     ax.scatter(
        #         [q_med],
        #         [sk_med],
        #         [marker_z],
        #         marker="X",
        #         s=130,
        #         color="black",
        #         depthshade=False,
        #         label="Median structure",
        #         zorder=20,
        #     )
        #     ax.text(
        #         q_med,
        #         sk_med,
        #         marker_z + 0.002,
        #         "Median",
        #         color="black",
        #         fontsize=10,
        #         ha="left",
        #         va="bottom",
        #         zorder=21,
        #     )
        # ax.legend(
        #     loc="upper left",
        #     bbox_to_anchor=(-0.10, 0.88),
        #     frameon=True,
        #     edgecolor="black",
        #     fontsize=11,
        # )
        # ========================================================
        # LAYOUT
        # ========================================================
        plt.tight_layout()
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(-0.10, 0.88),
            frameon=True,
            edgecolor="black",
            fontsize=11,
        )
        # ========================================================
        # SAVE
        # ========================================================
        outfile = os.path.join(
            outdir,
            f"{safe}_{condition}_"
            f"q_vs_Sk_population_3D_median_LSI.png"
        )
        plt.savefig(
            outfile,
            dpi=300,
            bbox_inches="tight",
        )
        print(
            f"Saved: {outfile}"
        )
        plt.close()

def load_depth_resolved_sq(
    path,
    condition,
):
    """Depth-resolved in-plane structure factor of one condition.

    Parameters
    ----------
    path : 
        Input file or directory.
    condition : 
        Liquid or glass label, Cool or Warm.

    Returns
    -------
    result
        Depth-resolved in-plane structure factor of one condition.
    """
    sq_file = os.path.join(
        path,
        ALL_CONDITIONS[
            condition
        ]["sq_file"]
    )
    if not os.path.exists(
        sq_file
    ):
        print(
            f"Missing {condition} "
            f"depth-resolved S(q): "
            f"{sq_file}"
        )
        return None
    df = pd.read_csv(
        sq_file
    )
    required_columns = [
        "depth_A",
        "q_Ainv",
        "Sq_mean",
        "Sq_sd",
        "gibbs_abs_z_A",
    ]
    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]
    if missing:
        raise RuntimeError(
            f"{sq_file} is missing "
            f"required columns: "
            f"{missing}"
        )
    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )
    df = df.dropna(
        subset=[
            "depth_A",
            "q_Ainv",
            "Sq_mean",
        ]
    )
    print(
        f"Loaded {condition} "
        f"depth-resolved S(q): "
        f"{sq_file}"
    )
    print(
        f"  Gibbs surface: "
        f"{df['gibbs_abs_z_A'].iloc[0]:.3f} A"
    )
    print(
        "  Depths:",
        np.sort(
            df["depth_A"].unique()
        )
    )
    return df

def plot_depth_resolved_sq(
    condition,
):
    # ========================================================
    # USER PLOT LIMIT
    # ========================================================
    """In-plane structure factor at each depth.

    Parameters
    ----------
    condition : 
        Liquid or glass label, Cool or Warm.

    Returns
    -------
    result
        In-plane structure factor at each depth.
    """
    MAX_DEPTH_A = 60.0
    for model_key, model_info in models.items():
        display = model_info[
            "display"
        ]
        safe = model_info[
            "safe"
        ]
        path = model_info[
            "path"
        ]
        df = load_depth_resolved_sq(
            path=path,
            condition=condition,
        )
        if (
            df is None
            or
            len(df) == 0
        ):
            continue
        # ====================================================
        # GIBBS DIVIDING SURFACE
        # ====================================================
        gibbs_abs_z = float(
            df[
                "gibbs_abs_z_A"
            ].iloc[0]
        )
        print(
            f"\n{display} {condition}: "
            f"depth-resolved S_parallel(q)"
        )
        print(
            f"|z_G| = "
            f"{gibbs_abs_z:.3f} A"
        )
        # ====================================================
        # RESTRICT ANALYSIS TO |z| <= 60 A
        # ====================================================
        df_plot = df[
            df["depth_A"]
            <= MAX_DEPTH_A + 1.0e-6
        ].copy()
        if len(df_plot) == 0:
            print(
                f"No S(q) data found for "
                f"|z| <= {MAX_DEPTH_A:.1f} A"
            )
            continue
        depths = np.sort(
            df_plot[
                "depth_A"
            ].unique()
        )
        outermost_depth = float(
            np.max(
                depths
            )
        )
        print(
            f"Plotting depths through "
            f"|z| = {outermost_depth:.1f} A"
        )
        print(
            f"Gibbs surface lies at "
            f"|z_G| = {gibbs_abs_z:.1f} A"
        )
        # ====================================================
        # COLOR SCALE
        # ====================================================
        #
        # IMPORTANT:
        #
        # Use exactly 0 -> 60 A for both 78 K and 300 K.
        #
        # This ensures identical colors correspond to
        # identical physical depths in both figures.
        # ====================================================
        depth_norm = Normalize(
            vmin=0.0,
            vmax=MAX_DEPTH_A,
        )
        depth_cmap = plt.get_cmap(
            "viridis"
        )
        # ====================================================
        # FIGURE
        # ====================================================
        fig, ax = plt.subplots(
            figsize=(
                10.5,
                7.0,
            )
        )
        # ====================================================
        # PLOT EACH DEPTH
        # ====================================================
        for depth in depths:
            sub = (
                df_plot[
                    np.isclose(
                        df_plot[
                            "depth_A"
                        ],
                        depth,
                    )
                ]
                .sort_values(
                    "q_Ainv"
                )
            )
            q_values = (
                sub[
                    "q_Ainv"
                ].to_numpy()
            )
            sq_mean = (
                sub[
                    "Sq_mean"
                ].to_numpy()
            )
            sq_sd = (
                sub[
                    "Sq_sd"
                ].to_numpy()
            )
            color = depth_cmap(
                depth_norm(
                    depth
                )
            )
            # ------------------------------------------------
            # Identify special curves
            # ------------------------------------------------
            is_center = np.isclose(
                depth,
                0.0,
            )
            is_outermost = np.isclose(
                depth,
                outermost_depth,
                atol=1.0e-3,
            )
            if is_center:
                linewidth = 3
                linestyle = "-"
            elif is_outermost:
                linewidth = 3
                linestyle = "--"
            else:
                linewidth = 1.8
                linestyle = "-"
            # ------------------------------------------------
            # Main curve
            # ------------------------------------------------
            ax.plot(
                q_values,
                sq_mean,
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=0.950,
                zorder=(
                    10
                    if is_center
                    or is_outermost
                    else 5
                ),
            )
            # ------------------------------------------------
            # Frame-to-frame width of S(q), not a replica
            # uncertainty. Shown for the center and the
            # outermost near-interface slice.
            # ------------------------------------------------
            if (
                is_center
                or
                is_outermost
            ):
                good_sd = np.isfinite(
                    sq_sd
                )
                if np.any(
                    good_sd
                ):
                    lower = (
                        sq_mean
                        - sq_sd
                    )
                    upper = (
                        sq_mean
                        + sq_sd
                    )
                    ax.fill_between(
                        q_values,
                        lower,
                        upper,
                        color=color,
                        alpha=0.12,
                        linewidth=0,
                        label="Frame-to-frame width" if is_center else None,
                    )
        # ====================================================
        # MANUAL LABELS FOR SPECIAL CURVES
        # ====================================================
        center_color = depth_cmap(
            depth_norm(
                0.0
            )
        )
        outer_color = depth_cmap(
            depth_norm(
                outermost_depth
            )
        )
        ax.plot(
            [],
            [],
            color=center_color,
            linewidth=3,
            linestyle="-",
            label=r"Slab Center ($z=0$)",
        )
        ax.plot(
            [],
            [],
            color=outer_color,
            linewidth=3,
            linestyle="--",
            label=(
                rf"Near Interface "
                rf"($|z|="
                rf"{outermost_depth:.0f}$ Å)"
            ),
        )
        # ====================================================
        # AXES
        # ====================================================
        ax.set_xlabel(
            r"Wave Vector, "
            r"$q$ ($\mathrm{\AA}^{-1}$)",
            fontsize=15,
        )
        ax.set_ylabel(
            r"In-Plane Oxygen Structure Factor, "
            r"$S_{\parallel}(q)$",
            fontsize=15,
        )
        # Keep identical q range between 78 K and 300 K
        ax.set_xlim(
            0.5,
            5.0,
        )
        # Keep identical S_parallel(q) range between 78 K and 300 K
        ax.set_ylim(
            0.0,
            2,
        )
        ax.grid(
            linestyle="--",
            alpha=0.250,
        )
        ax.tick_params(
            axis="both",
            labelsize=12,
        )
        # ====================================================
        # TITLE
        # ====================================================
        ax.set_title(
            rf"{display} Depth-Resolved "
            rf"In-Plane Structure Factor at "
            rf"{condition_temp(condition)}"
            "\n"
            rf"$z=0,\ \pm5,\ \pm10,\ldots,"
            rf"\ \pm60$ Å"
            rf"   ($|z_G|\approx"
            rf"{gibbs_abs_z:.0f}$ Å)",
            fontsize=16,
            fontweight="bold",
            pad=14,
        )
        # ====================================================
        # DEPTH COLORBAR
        # ====================================================
        sm = ScalarMappable(
            norm=depth_norm,
            cmap=depth_cmap,
        )
        sm.set_array([])
        cbar = fig.colorbar(
            sm,
            ax=ax,
            pad=0.0250,
            fraction=0.045,
        )
        cbar.set_label(
            r"Distance from Slab Center, "
            r"$|z|$ (Å)",
            fontsize=13,
        )
        # Fixed depth ticks: directly comparable between
        # 78 K and 300 K.
        depth_ticks = np.arange(
            0.0,
            MAX_DEPTH_A + 0.1,
            10.0,
        )
        cbar.set_ticks(
            depth_ticks
        )
        # ====================================================
        # LEGEND
        # ====================================================
        ax.legend(
            loc="best",
            frameon=True,
            edgecolor="black",
            fontsize=11,
        )
        # ====================================================
        # SAVE
        # ====================================================
        plt.tight_layout()
        outfile = os.path.join(
            outdir,
            f"{safe}_{condition}_"
            f"Sq2D_depth_resolved_0to60A.png"
        )
        plt.savefig(
            outfile,
            dpi=300,
            bbox_inches="tight",
        )
        print(
            f"Saved: {outfile}"
        )
        plt.close()

def plot_depth_resolved_sq_cool_vs_warm():
    """Liquid and glass structure factors on the same axes.

    Returns
    -------
    result
        Liquid and glass structure factors on the same axes.
    """
    for model_key, model_info in models.items():
        display = model_info[
            "display"
        ]
        safe = model_info[
            "safe"
        ]
        path = model_info[
            "path"
        ]
        cool = load_depth_resolved_sq(
            path,
            "Cool",
        )
        warm = load_depth_resolved_sq(
            path,
            "Warm",
        )
        if (
            cool is None
            or
            warm is None
        ):
            print(
                "Both Cool and Warm "
                "S(q) files are required "
                "for comparison."
            )
            continue
        # ====================================================
        # COMMON SCALE
        # ====================================================
        cool_gibbs = float(
            cool[
                "gibbs_abs_z_A"
            ].iloc[0]
        )
        warm_gibbs = float(
            warm[
                "gibbs_abs_z_A"
            ].iloc[0]
        )
        max_gibbs = max(
            cool_gibbs,
            warm_gibbs,
        )
        depth_norm = Normalize(
            vmin=0.0,
            vmax=max_gibbs,
        )
        depth_cmap = plt.get_cmap(
            "viridis"
        )
        # ----------------------------------------------------
        # Shared y-range
        # ----------------------------------------------------
        all_sq = np.concatenate(
            [
                cool[
                    "Sq_mean"
                ].to_numpy(),
                warm[
                    "Sq_mean"
                ].to_numpy(),
            ]
        )
        finite_sq = all_sq[
            np.isfinite(
                all_sq
            )
        ]
        y_min = max(
            0.0,
            np.min(
                finite_sq
            )
            - 0.05
        )
        y_max = (
            np.max(
                finite_sq
            )
            * 1.05
        )
        # ====================================================
        # FIGURE
        # ====================================================
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(
                15,
                6.5,
            ),
            sharex=True,
            sharey=True,
        )
        condition_data = [
            (
                "Warm",
                warm,
                warm_gibbs,
            ),
            (
                "Cool",
                cool,
                cool_gibbs,
            ),
        ]
        for (
            ax,
            (
                condition,
                data,
                gibbs_z,
            )
        ) in zip(
            axes,
            condition_data,
        ):
            depths = np.sort(
                data[
                    "depth_A"
                ].unique()
            )
            for depth in depths:
                sub = (
                    data[
                        np.isclose(
                            data[
                                "depth_A"
                            ],
                            depth,
                        )
                    ]
                    .sort_values(
                        "q_Ainv"
                    )
                )
                color = depth_cmap(
                    depth_norm(
                        depth
                    )
                )
                is_center = np.isclose(
                    depth,
                    0.0,
                )
                is_gibbs = np.isclose(
                    depth,
                    gibbs_z,
                    atol=1.0e-3,
                )
                if is_center:
                    linewidth = 3
                    linestyle = "-"
                elif is_gibbs:
                    linewidth = 3
                    linestyle = "--"
                else:
                    linewidth = 1.6
                    linestyle = "-"
                ax.plot(
                    sub[
                        "q_Ainv"
                    ],
                    sub[
                        "Sq_mean"
                    ],
                    color=color,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    alpha=0.950,
                )
            ax.set_title(
                rf"{condition_temp(condition)}"
                "\n"
                rf"$|z_G|="
                rf"{gibbs_z:.1f}$ Å",
                fontsize=15,
                fontweight="bold",
            )
            ax.set_xlabel(
                r"$q$ "
                r"($\mathrm{\AA}^{-1}$)",
                fontsize=14,
            )
            ax.set_xlim(
                0.5,
                5.0,
            )
            ax.set_ylim(
                y_min,
                y_max,
            )
            ax.grid(
                linestyle="--",
                alpha=0.250,
            )
        axes[0].set_ylabel(
            r"In-Plane Oxygen Structure Factor, "
            r"$S_{\parallel}(q)$",
            fontsize=14,
        )
        # ====================================================
        # SHARED DEPTH COLORBAR
        # ====================================================
        sm = ScalarMappable(
            norm=depth_norm,
            cmap=depth_cmap,
        )
        sm.set_array([])
        cbar = fig.colorbar(
            sm,
            ax=axes,
            pad=0.02,
            fraction=0.035,
        )
        cbar.set_label(
            r"Distance from Slab Center, "
            r"$|z|$ (Å)",
            fontsize=13,
        )
        # ====================================================
        # OVERALL TITLE
        # ====================================================
        fig.suptitle(
            rf"{display} Depth-Resolved "
            rf"In-Plane Structure Factor:"
            "\n"
            rf"300 K vs 78 K",
            fontsize=17,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout(
            rect=[
                0.0,
                0.0,
                0.92,
                0.92,
            ]
        )
        outfile = os.path.join(
            outdir,
            f"{safe}_Warm_vs_Cool_"
            f"Sq2D_depth_resolved.png"
        )
        plt.savefig(
            outfile,
            dpi=300,
            bbox_inches="tight",
        )
        print(
            f"Saved: {outfile}"
        )
        plt.close()

def bragg_output_files(path, condition):
    """Expected Bragg-analysis files produced by the extraction script.

    Parameters
    ----------
    path : 
        Input file or directory.
    condition : 
        Liquid or glass label, Cool or Warm.

    Returns
    -------
    result
        Expected Bragg-analysis files produced by the extraction script.
    """
    prefix = "warm_" if condition == "Warm" else ""
    return {
        "map": os.path.join(path, f"{prefix}water_Sq2D_center_bragg_map.npz"),
        "peaks": os.path.join(path, f"{prefix}water_bragg_peak_diagnostics.csv"),
        "angular": os.path.join(path, f"{prefix}water_bragg_angular_anisotropy.csv"),
    }

def load_bragg_outputs(path, condition):
    """Bragg map, radial peaks, and angular anisotropy, or None if a file is missing.

    Parameters
    ----------
    path : 
        Input file or directory.
    condition : 
        Liquid or glass label, Cool or Warm.

    Returns
    -------
    result
        Bragg map, radial peaks, and angular anisotropy, or None if a file is missing.
    """
    files = bragg_output_files(path, condition)
    missing = [f for f in files.values() if not os.path.exists(f)]
    if missing:
        print(f"\nMissing {condition} Bragg-analysis files:")
        for f in missing:
            print(f"  {f}")
        return None
    with np.load(files["map"]) as npz:
        bragg_map = {key: np.array(npz[key]) for key in npz.files}
    peaks = pd.read_csv(files["peaks"])
    angular = pd.read_csv(files["angular"])
    print(f"\nLoaded {condition} Bragg diagnostics")
    print(f"  map:     {files['map']}")
    print(f"  peaks:   {files['peaks']}")
    print(f"  angular: {files['angular']}")
    return {"map": bragg_map, "peaks": peaks, "angular": angular}

def plot_bragg_reciprocal_map(condition):
    """Plot central-slab time-averaged S(qx,qy).

    Parameters
    ----------
    condition : 
        Liquid or glass label, Cool or Warm.

    Returns
    -------
    result
        Plot central-slab time-averaged S(qx,qy).
    """
    for _, model_info in models.items():
        display = model_info["display"]
        safe = model_info["safe"]
        path = model_info["path"]
        data = load_bragg_outputs(path, condition)
        if data is None:
            continue
        m = data["map"]
        if not all(k in m for k in ["Sq_mean", "qx", "qy"]):
            print(f"Incomplete Bragg map for {display} {condition}")
            continue
        sq = np.asarray(m["Sq_mean"], dtype=float)
        qx = np.asarray(m["qx"], dtype=float)
        qy = np.asarray(m["qy"], dtype=float)
        sq = np.fft.fftshift(sq, axes=(0, 1))
        qx = np.fft.fftshift(qx)
        qy = np.fft.fftshift(qy)
        qlim = 5.0
        mx = (qx >= -qlim) & (qx <= qlim)
        my = (qy >= -qlim) & (qy <= qlim)
        sq = sq[np.ix_(mx, my)]
        qx = qx[mx]
        qy = qy[my]
        positive = sq[np.isfinite(sq) & (sq > 0)]
        if len(positive) == 0:
            print(f"No finite Bragg-map intensities for {display} {condition}")
            continue
        floor = max(np.nanpercentile(positive, 1.0), np.finfo(float).tiny)
        log_sq = np.log10(np.where(sq > 0, sq, floor))
        vmin = np.nanpercentile(log_sq, 2.0)
        vmax = np.nanpercentile(log_sq, 99.7)
        fig, ax = plt.subplots(figsize=(8.5, 7.5))
        im = ax.pcolormesh(
            qx, qy, log_sq.T,
            shading="auto", cmap="viridis",
            vmin=-0.7, vmax=0.7,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-qlim, qlim)
        ax.set_ylim(-qlim, qlim)
        ax.set_xlabel(r"$q_x$ ($\mathrm{\AA}^{-1}$)", fontsize=15)
        ax.set_ylabel(r"$q_y$ ($\mathrm{\AA}^{-1}$)", fontsize=15)
        ax.set_title(
            rf"{display} Central-Slab Reciprocal-Space Map at {condition_temp(condition)}"
            "\n" rf"$\log_{{10}} S(q_x,q_y)$",
            fontsize=16, fontweight="bold", pad=14,
        )
        ax.grid(linestyle="--", alpha=0.18)
        cbar = fig.colorbar(im, ax=ax, pad=0.0250, fraction=0.046)
        cbar.set_label(r"$\log_{10} S(q_x,q_y)$", fontsize=13)
        if "dq_x_resolution_Ainv" in m and "dq_y_resolution_Ainv" in m:
            dqx = float(np.asarray(m["dq_x_resolution_Ainv"]))
            dqy = float(np.asarray(m["dq_y_resolution_Ainv"]))
            ax.text(
                0.02, 0.98,
                rf"$\Delta q_x={dqx:.4f}$ Å$^{{-1}}$" "\n"
                rf"$\Delta q_y={dqy:.4f}$ Å$^{{-1}}$",
                transform=ax.transAxes, ha="left", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="black", alpha=0.9),
            )
        plt.tight_layout()
        outfile = os.path.join(outdir, f"{safe}_{condition}_Bragg_Sq2D_map.png")
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        print(f"Saved: {outfile}")
        plt.close()

def plot_bragg_radial_peaks(condition):
    """Plot z=0 radial S_parallel(q), detected peaks, and their FWHM.

    Parameters
    ----------
    condition : 
        Liquid or glass label, Cool or Warm.

    Returns
    -------
    result
        Plot z=0 radial S_parallel(q), detected peaks, and their FWHM.
    """
    for _, model_info in models.items():
        display = model_info["display"]
        safe = model_info["safe"]
        path = model_info["path"]
        data = load_bragg_outputs(path, condition)
        if data is None:
            continue
        peaks = data["peaks"].copy()
        sq_df = load_depth_resolved_sq(path, condition)
        if sq_df is None or len(sq_df) == 0:
            continue
        center = sq_df[np.isclose(sq_df["depth_A"], 0.0)].sort_values("q_Ainv")
        if len(center) == 0:
            print(f"No z=0 S(q) curve for {display} {condition}")
            continue
        q = center["q_Ainv"].to_numpy()
        s = center["Sq_mean"].to_numpy()
        sd = center["Sq_sd"].to_numpy() if "Sq_sd" in center.columns else None
        if "candidate_bragg_peak" in peaks.columns:
            flag_col = "candidate_bragg_peak"
        elif "narrow_peak_flag" in peaks.columns:
            flag_col = "narrow_peak_flag"
        else:
            flag_col = None
        flags = peaks[flag_col].astype(bool).to_numpy() if flag_col else np.zeros(len(peaks), dtype=bool)
        fig, ax = plt.subplots(figsize=(10.5, 7.0))
        ax.plot(q, s, linewidth=2.6, label=r"Central slab $S_{\parallel}(q)$")
        if sd is not None and np.any(np.isfinite(sd)):
            ax.fill_between(q, s - sd, s + sd, alpha=0.12, linewidth=0, label="Frame-to-frame width")
        for j, (_, row) in enumerate(peaks.iterrows()):
            q0 = float(row["q_peak_Ainv"])
            s0 = float(row["Sq_peak"])
            fwhm = float(row["FWHM_Ainv"]) if "FWHM_Ainv" in row and np.isfinite(row["FWHM_Ainv"]) else np.nan
            candidate = bool(flags[j])
            ax.scatter([q0], [s0], marker="*" if candidate else "o",
                       s=150 if candidate else 55, edgecolor="black", linewidth=0.7, zorder=15)
            if np.isfinite(fwhm):
                ax.hlines(s0, q0 - 0.5*fwhm, q0 + 0.5*fwhm, linewidth=2.5, zorder=14)
            if candidate:
                ax.annotate(rf"$q={q0:.3f}$", (q0, s0), xytext=(6,8),
                            textcoords="offset points", fontsize=9, fontweight="bold")
        if len(peaks) > 0:
            ax.scatter([], [], marker="o", s=55, edgecolor="black", linewidth=0.7,
                       label="Detected radial maximum")
        if np.any(flags):
            ax.scatter([], [], marker="*", s=150, edgecolor="black", linewidth=0.7,
                       label="Candidate narrow / Bragg-like peak")
        if len(peaks) > 0 and "dq_resolution_Ainv" in peaks.columns:
            vals = peaks["dq_resolution_Ainv"].dropna()
            if len(vals) > 0:
                dq = float(vals.iloc[0])
                ax.text(0.02, 0.98,
                        rf"Reciprocal resolution: $\Delta q\approx{dq:.4f}$ Å$^{{-1}}$",
                        transform=ax.transAxes, ha="left", va="top", fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="black", alpha=0.9))
        ax.set_xlabel(r"Wave Vector, $q$ ($\mathrm{\AA}^{-1}$)", fontsize=15)
        ax.set_ylabel(r"Central-Slab $S_{\parallel}(q)$", fontsize=15)
        ax.set_xlim(1.0, 5.0)
        ax.set_ylim(bottom=0.0)
        ax.set_title(
            rf"{display} Bragg-Peak Diagnostic at {condition_temp(condition)}"
            "\n" r"Central Slab ($z=0$)",
            fontsize=16, fontweight="bold", pad=14,
        )
        ax.grid(linestyle="--", alpha=0.30)
        ax.legend(frameon=True, edgecolor="black", loc="best", fontsize=10)
        plt.tight_layout()
        outfile = os.path.join(outdir, f"{safe}_{condition}_Bragg_radial_peak_diagnostics.png")
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        print(f"Saved: {outfile}")
        plt.close()

def plot_bragg_angular_anisotropy(condition):
    """Plot A2/A4/A6 and spot-to-mean ring intensity for detected peaks.

    Parameters
    ----------
    condition : 
        Liquid or glass label, Cool or Warm.

    Returns
    -------
    result
        Plot A2/A4/A6 and spot-to-mean ring intensity for detected peaks.
    """
    for _, model_info in models.items():
        display = model_info["display"]
        safe = model_info["safe"]
        path = model_info["path"]
        data = load_bragg_outputs(path, condition)
        if data is None:
            continue
        angular = data["angular"].replace([np.inf, -np.inf], np.nan).copy()
        if len(angular) == 0 or "q_peak_Ainv" not in angular.columns:
            print(f"No angular Bragg data to plot for {display} {condition}")
            continue
        angular = angular.dropna(subset=["q_peak_Ainv"]).sort_values("q_peak_Ainv")
        if len(angular) == 0:
            continue
        series = [(angular, condition_temp(condition), "-")]
        if condition == "Cool":
            warm_data = load_bragg_outputs(path, "Warm")
            if warm_data is None:
                print(f"Skipping {display} at 78 K: A_m is not shown without the 300 K baseline.")
                continue
            warm = warm_data["angular"].replace([np.inf, -np.inf], np.nan).dropna(subset=["q_peak_Ainv"]).sort_values("q_peak_Ainv")
            series.append((warm, condition_temp("Warm"), "--"))
        fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.2), sharex=True,
                                 gridspec_kw={"height_ratios": [2.0, 1.2]})
        ax1, ax2 = axes
        plotted = False
        for table, label_t, style in series:
            for col, label, marker in [("A2", r"$A_2$", "o"), ("A4", r"$A_4$", "s"), ("A6", r"$A_6$", "^")]:
                if col in table.columns:
                    ax1.plot(table["q_peak_Ainv"], table[col], marker=marker, linestyle=style,
                             linewidth=2.0, markersize=6, label=f"{label} {label_t}")
                    plotted = True
        ax1.set_ylabel("Angular Anisotropy", fontsize=14)
        ax1.set_ylim(0.0, 1.0)
        ax1.grid(linestyle="--", alpha=0.30)
        if plotted:
            ax1.legend(frameon=True, edgecolor="black", loc="best")
        for table, label_t, style in series:
            if "spot_to_mean_ratio" in table.columns:
                ax2.plot(table["q_peak_Ainv"], table["spot_to_mean_ratio"], marker="o", linestyle=style, linewidth=2.2, label=label_t)
        ax2.axhline(1.0, linestyle="--", linewidth=1.0, alpha=0.6)
        ax2.set_xlabel(r"Detected Radial-Peak Position, $q$ ($\mathrm{\AA}^{-1}$)", fontsize=14)
        ax2.set_ylabel("Spot / Mean\nRing Intensity", fontsize=13)
        ax2.grid(linestyle="--", alpha=0.30)
        fig.suptitle(
            rf"{display} Reciprocal-Space Angular Diagnostics at {condition_temp(condition)}"
            "\n" r"Central Slab ($z=0$)",
            fontsize=16, fontweight="bold", y=0.98,
        )
        plt.tight_layout(rect=[0,0,1,0.94])
        outfile = os.path.join(outdir, f"{safe}_{condition}_Bragg_angular_anisotropy.png")
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        print(f"Saved: {outfile}")
        plt.close()

def plot_bragg_diagnostics(condition):
    """Generate all Bragg/crystallinity plots for one condition.

    Parameters
    ----------
    condition : 
        Liquid or glass label, Cool or Warm.

    Returns
    -------
    result
        Generate all Bragg/crystallinity plots for one condition.
    """
    plot_bragg_reciprocal_map(condition)
    plot_bragg_radial_peaks(condition)
    plot_bragg_angular_anisotropy(condition)

def main(argv=None):
    """Write the local-structure figures for the requested conditions.

    Parameters
    ----------
    argv : 
        Arguments excluding the program name. Defaults to sys.argv.

    Returns
    -------
    result
        Write the local-structure figures for the requested conditions.
    """
    global RUN_MODE, ALL_CONDITIONS, run_mode, ACTIVE_CONDITIONS, conditions, condition, models, outdir, model_colors, condition_temperature_labels, profiles, model, model_info, display, safe, path, cfg, f, model_key, q_file, sk_file, lsi_file, chill_file
    # ============================================================
    # USER SETTINGS
    # ============================================================
    # ============================================================
    # CONDITION SELECTION
    # ============================================================
    # Allowed values:
    #     "cool"
    #     "warm"
    #     "both"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Directory of local-order outputs.")
    parser.add_argument("--condition", choices=["cool", "warm", "both"], default="both")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args(argv)
    RUN_MODE = args.condition
    ALL_CONDITIONS = {
        "Warm": {
            "profile_file":
                "warm_water_local_order_z_profile_10A.csv",
            "ptheta_file":
                "warm_water_angular_distribution_Ptheta_by_z_window.csv",
            "sq_file":
                "warm_water_Sq2D_by_depth.csv",
            "linestyle":
                "--",
        },
        "Cool": {
            "profile_file":
                "water_local_order_z_profile_10A.csv",
            "ptheta_file":
                "water_water_angular_distribution_Ptheta_by_z_window.csv",
            "sq_file":
                "water_Sq2D_by_depth.csv",
            "linestyle":
                "-",
        },
    }
    run_mode = RUN_MODE.strip().lower()
    if run_mode == "cool":
        ACTIVE_CONDITIONS = ["Cool"]
    elif run_mode == "warm":
        ACTIVE_CONDITIONS = ["Warm"]
    elif run_mode == "both":
        ACTIVE_CONDITIONS = ["Warm", "Cool"]
    else:
        raise ValueError(
            f"Invalid RUN_MODE='{RUN_MODE}'. "
            f"Choose 'cool', 'warm', or 'both'."
        )
    # Keep the rest of the script compatible with the existing
    # variable name \`conditions\`.
    conditions = {
        condition: ALL_CONDITIONS[condition]
        for condition in ACTIVE_CONDITIONS
    }
    print(
        "Running conditions: "
        + ", ".join(ACTIVE_CONDITIONS)
    )
    models = {
        "OPC": {
            "display": "OPC",
            "safe": "OPC",
            "path": args.data_dir,
        },
    }
    outdir = args.out
    os.makedirs(outdir, exist_ok=True)
    model_colors = {
        "OPC": "black",
        "TIP3P": "tab:blue",
        "TIP4P": "tab:orange",
        "SPC/E": "tab:green",
    }
    plt.rcParams.update({
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 17,
        "axes.titleweight": "bold",
        "legend.fontsize": 10,
    })
    condition_temperature_labels = {
        "Warm": "300 K",
        "Cool": "78 K",
    }
    # ============================================================
    # LOAD
    # ============================================================
    profiles = {}
    for model, model_info in models.items():
        display = model_info["display"]
        safe = model_info["safe"]
        path = model_info["path"]
        profiles[model] = {}
        for condition, cfg in conditions.items():
            f = os.path.join(path, cfg["profile_file"])
            if os.path.exists(f):
                profiles[model][condition] = pd.read_csv(f)
                print(f"Loaded {display} {condition}: {f}")
            else:
                print(f"Missing {display} {condition}: {f}")
    # ============================================================
    # PROFILE PLOTTER: WARM VS COOL FOR EACH MODEL
    # ============================================================
    # ============================================================
    # DIFFERENCE PLOTTER: COOL - WARM
    # ============================================================
    # ============================================================
    # MAIN Z-PROFILES: WARM VS COOL
    # ============================================================
    # plot_z_profile_pairwise(
    #     column="mean_q_orientational_tetrahedral",
    #     sd_column="sd_q_orientational_tetrahedral",
    #     xlabel=r"Orientational Tetrahedral Order $\langle q \rangle$",
    #     filename="warm_vs_cool_orientational_tetrahedral_q_z.png",
    #     title=r"OPC Warm vs Cool: Orientational Tetrahedral Order",
    # )
    # plot_z_profile_pairwise(
    #     column="mean_Sk_translational_tetrahedral",
    #     sd_column="sd_Sk_translational_tetrahedral",
    #     xlabel=r"Translational Tetrahedral Order $\langle S_k \rangle$",
    #     filename="warm_vs_cool_translational_tetrahedral_Sk_z.png",
    #     title=r"OPC Warm vs Cool: Translational Tetrahedral Order",
    # )
    # plot_z_profile_pairwise(
    #     column="mean_LSI",
    #     sd_column="sd_LSI",
    #     xlabel=r"Local Structure Index $\langle \mathrm{LSI} \rangle$",
    #     filename="warm_vs_cool_LSI_z.png",
    #     title=r"OPC Warm vs Cool: Local Structure Index",
    # )
    # ============================================================
    # DELTA PROFILES: COOL - WARM
    # ============================================================
    # plot_delta_profile(
    #     column="mean_q_orientational_tetrahedral",
    #     xlabel=r"$\Delta \langle q \rangle$ = cool - warm",
    #     filename="delta_cool_minus_warm_q_z.png",
    #     title=r"OPC Cooling-Induced Change in Orientational Tetrahedral Order",
    # )
    # plot_delta_profile(
    #     column="mean_Sk_translational_tetrahedral",
    #     xlabel=r"$\Delta \langle S_k \rangle$ = cool - warm",
    #     filename="delta_cool_minus_warm_Sk_z.png",
    #     title=r"OPC Cooling-Induced Change in Translational Tetrahedral Order",
    # )
    # plot_delta_profile(
    #     column="mean_LSI",
    #     xlabel=r"$\Delta \langle \mathrm{LSI} \rangle$ = cool - warm",
    #     filename="delta_cool_minus_warm_LSI_z.png",
    #     title=r"OPC Cooling-Induced Change in Local Structure Index",
    # )
    # def plot_q_sk_lsi_median_landscape(condition):
    #     q_bins = np.linspace(-0.2, 1.0, 80)
    #     sk_bins = np.linspace(0.0, 1.0, 80)
    #     vmin = 0.0
    #     vmax = 0.20
    #     for model_key, model_info in models.items():
    #         display = model_info["display"]
    #         safe = model_info["safe"]
    #         path = model_info["path"]
    #         df = load_q_sk_lsi_combined(path, condition)
    #         if df is None or len(df) == 0:
    #             print(f"Missing q/Sk/LSI combined data for {display} {condition}")
    #             continue
    #         q = df["q"].to_numpy()
    #         sk = df["Sk"].to_numpy()
    #         lsi = df["LSI"].to_numpy()
    #         median_lsi, q_edges, sk_edges, _ = binned_statistic_2d(
    #             q,
    #             sk,
    #             lsi,
    #             statistic="median",
    #             bins=[q_bins, sk_bins],
    #         )
    #         counts, _, _, _ = binned_statistic_2d(
    #             q,
    #             sk,
    #             lsi,
    #             statistic="count",
    #             bins=[q_bins, sk_bins],
    #         )
    #         median_lsi[counts < 10] = np.nan
    #         fig, ax = plt.subplots(figsize=(7.2, 6.0))
    #         im = ax.imshow(
    #             median_lsi.T,
    #             origin="lower",
    #             aspect="auto",
    #             extent=[
    #                 q_bins.min(),
    #                 q_bins.max(),
    #                 sk_bins.min(),
    #                 sk_bins.max(),
    #             ],
    #             cmap="turbo_r",
    #             vmin=0.00,
    #             vmax=0.20,
    #         )
    #         q_centers = 0.5 * (q_edges[:-1] + q_edges[1:])
    #         sk_centers = 0.5 * (sk_edges[:-1] + sk_edges[1:])
    #         ax.contour(
    #             q_centers,
    #             sk_centers,
    #             counts.T,
    #             levels=5,
    #             colors="black",
    #             linewidths=0.6,
    #             alpha=0.55,
    #         )
    #         ax.set_xlabel(r"Orientational Tetrahedral Order $q$")
    #         ax.set_ylabel(r"Translational Tetrahedral Order $S_k$")
    #         ax.set_title(
    #             make_plot_title(
    #                 display,
    #                 "Local Structure Landscape:\n"
    #                 rf"$q$ vs $S_k$ Colored by Median LSI, bulk |z| <= {BULK_HALFWIDTH_A:g} A",
    #                 condition,
    #             ),
    #             pad=16,
    #         )
    #         ax.set_xlim(-0.2, 1.0)
    #         ax.set_ylim(0.0, 1.0)
    #         ax.grid(linestyle="--", alpha=0.250)
    #         cbar = fig.colorbar(
    #             im,
    #             ax=ax,
    #             pad=0.03,
    #             fraction=0.046,
    #         )
    #         cbar.set_label(r"Median LSI ($\mathrm{\AA}^2$)")
    #         plt.tight_layout()
    #         plt.savefig(
    #             os.path.join(
    #                 outdir,
    #                 f"{safe}_{condition}_q_vs_Sk_median_LSI_landscape.png"
    #             ),
    #             dpi=300,
    #             bbox_inches="tight",
    #         )
    #         plt.close()
    # ============================================================
    # LOAD DEPTH-RESOLVED STRUCTURE FACTOR
    # ============================================================
    # ============================================================
    # DEPTH-RESOLVED S_parallel(q)
    # ============================================================
    # ============================================================
    # COOL VS WARM DEPTH-RESOLVED S(q)
    # ============================================================
    # ============================================================
    # BRAGG / CRYSTALLINITY PLOTTING
    # ============================================================
    for model_key, model_info in models.items():
        display = model_info["display"]
        safe = model_info["safe"]
        path = model_info["path"]
        for condition in conditions.keys():
            q_file = os.path.join(path, f"{condition.lower()}_water_q_per_water_distribution.csv")
            sk_file = os.path.join(path, f"{condition.lower()}_water_Sk_per_water_distribution.csv")
            lsi_file = os.path.join(path, f"{condition.lower()}_water_LSI_per_water_distribution.csv")
            chill_file = os.path.join(path, f"{condition.lower()}_water_CHILLplus_z_profile_10A.csv")
        #     if os.path.exists(q_file):
        #         q_df = pd.read_csv(q_file)
        #
        #         plot_distribution_by_z(
        #             df=q_df,
        #             value_col="q",
        #             xlabel=r"Orientational tetrahedral order $q$",
        #             filename=f"{safe}_{condition}_q_distribution_by_z.png",
        #             model=display,
        #             condition=condition,
        #             plot_name=r"Distribution of $q$ by Z-Window",
        #             xlim=(-0.2, 1.0),
        #             use_kde=True,
        #         )
        #
        #     if os.path.exists(sk_file):
        #         sk_df = pd.read_csv(sk_file)
        #
        #         plot_distribution_by_z(
        #             df=sk_df,
        #             value_col="Sk",
        #             xlabel=r"Translational tetrahedral order $S_k$",
        #             filename=f"{safe}_{condition}_Sk_distribution_by_z.png",
        #             model=display,
        #             condition=condition,
        #             plot_name=r"Distribution of $S_k$ by Z-Window",
        #             xlim=(0.0, 1.0),
        #             use_kde=True,
        #         )
        #
            # if os.path.exists(lsi_file):
            #     lsi_df = pd.read_csv(lsi_file)
            #
            #     plot_distribution_by_z(
            #         df=lsi_df,
            #         value_col="LSI",
            #         xlabel=r"Local structure index LSI ($\mathrm{\AA}^2$)",
            #         filename=f"{safe}_{condition}_LSI_distribution_by_z.png",
            #         model=display,
            #         condition=condition,
            #         plot_name="Distribution of LSI by Z-Window",
            #         xlim=(0.0, 0.25),
            #         use_kde=True,
            #     )
            #
            #     plot_chillplus_by_distance_from_center(
            #         path=chill_file,
            #         display_name=display,
            #         safe_name=safe,
            #         condition=condition,
            #     )
    # ============================================================
    # q vs Sk LANDSCAPE COLORED BY MEDIAN LSI
    # ============================================================
    for condition in ACTIVE_CONDITIONS:
        plot_q_sk_lsi_median_landscape(
            condition=condition
        )
        plot_depth_resolved_sq(
        condition=condition
        )
        plot_bragg_diagnostics(
            condition=condition
        )
    # Direct comparison only when both datasets are requested
    if (
        "Cool" in ACTIVE_CONDITIONS
        and
        "Warm" in ACTIVE_CONDITIONS
    ):
        plot_depth_resolved_sq_cool_vs_warm()


if __name__ == "__main__":
    main()
