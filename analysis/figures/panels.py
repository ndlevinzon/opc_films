#!/usr/bin/env python3
"""Manuscript and SI panels that no other script draws.

Every panel reads finished pipeline outputs under ``--results`` and writes a
PNG and a PDF under ``--out``. A panel whose inputs are missing or unreadable
is skipped: its reason goes to ``<out>/FAILED_panels.txt`` and the other
panels are still drawn. Names follow the manuscript (``Fig4a_...``) or the SI
label (``SI_...``).

Main text
    Fig1b   cooling protocols, T against time
    Fig4a-b density and dipole orientation across the film, 300 and 78 K
    Fig4c-d depth-resolved in-plane S(q), 300 and 78 K
    Fig4e-f in-plane S(qx, qy) of the film center, 300 and 78 K
    Fig5a-b q, Sk and LSI landscape of bulk water, 300 and 78 K
    Fig5c   cross sections of the holds colored by the display classifier
    Fig5d-e LDA-like fraction and free volume of bulk and AWI on the holds
    Fig6a-c F_s, C1 and C2 of bulk and AWI at 235, 215 and 195 K
    Fig7a   intermittent H-bond partner retention at 235, 215 and 195 K
    Fig7b   bulk - AWI retention at 320 ps
    Fig7e   tau_alpha against the H-bond retention time

SI
    SI_order_profiles, SI_ptheta, SI_chill, SI_sq_depth   (endpoints)
    SI_fv_radius        free volume against probe radius (holds)
    SI_conservation     mass conservation of the ramp profiles
    SI_roughness, SI_capillary   capillary-wave outputs of the holds

All curves follow ``opcfilm.plotting.conventions``: temperature axes run hot
to cold, legends sit outside the panels, bulk is solid and AWI dashed, and
two-series panels are black.
"""

from __future__ import annotations

import argparse
import re
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from opcfilm.plotting.conventions import install_figure_conventions
install_figure_conventions()  # temperature axes hot to cold; legends outside panels
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)

from opcfilm.catalog import RATE_COLOR_BY_VALUE, load_manifest, result_directory
from opcfilm.config import CONFIG
from opcfilm.stats.uncertainty import paired_student_interval

BULK, AWI, AWI_OUTER = CONFIG.regions.names
# The classifier names its 3-12 Å interface band explicitly.
CLASSIFIER_AWI = "awi_3_12"
BULK_HALFWIDTH_A = CONFIG.regions.bulk_halfwidth_A
TEMPERATURE_COLORS = {235.0: "#c0392b", 215.0: "0.45", 195.0: "#2c5aa0"}
RATE = 4.44
CHILL_TYPES = {
    0: "Other",
    1: "Hexagonal ice",
    2: "Cubic ice",
    3: "Interfacial ice",
    4: "Hydrate",
    5: "Interfacial hydrate",
}
PANELS = {}


def panel(name):
    """Register a panel function under its output name.

    Parameters
    ----------
    name : str
        Output stem, such as ``Fig4a``.

    Returns
    -------
    callable
        Decorator that records the function.
    """
    def register(function):
        PANELS[name] = function
        return function
    return register


def save(figure, out, stem):
    """Write a figure as PNG and PDF and close it.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure to write.
    out : pathlib.Path
        Output directory.
    stem : str
        File name without extension.

    Returns
    -------
    list of pathlib.Path
        Files written.
    """
    out.mkdir(parents=True, exist_ok=True)
    files = [out / f"{stem}.png", out / f"{stem}.pdf"]
    figure.savefig(files[0], dpi=300)
    figure.savefig(files[1])
    plt.close(figure)
    return files


def need(path):
    """Return a path that must exist.

    Parameters
    ----------
    path : pathlib.Path
        Input file.

    Returns
    -------
    pathlib.Path
        The same path.

    Raises
    ------
    FileNotFoundError
        If the file is missing.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"missing input {path}")
    return Path(path)


def manifest_rows(results):
    """Manifest rows of the run whose outputs are under ``results``.

    Parameters
    ----------
    results : pathlib.Path
        The ``results`` directory. The manifest is ``../config/manifest.tsv``.

    Returns
    -------
    list of dict
        Manifest rows with the loader's aliases.
    """
    return load_manifest(need(results.parent / "config" / "manifest.tsv"))


def endpoint_dirs(results, temperature, stage="endpoint_structure"):
    """Output directories of the endpoints at one temperature.

    Parameters
    ----------
    results : pathlib.Path
        The ``results`` directory.
    temperature : float
        300 or 78.
    stage : str
        Stage directory under ``results``.

    Returns
    -------
    list of tuple
        ``(replica, directory)`` for each endpoint row that has outputs.
    """
    found = []
    for row in manifest_rows(results):
        if row["kind"] != "endpoint" or abs(float(row["temperature_K"]) - temperature) > 0.5:
            continue
        directory = result_directory(results.parent, stage, row)
        if directory.is_dir():
            found.append((row["replica"], directory))
    if not found:
        raise FileNotFoundError(f"no {stage} outputs for the {temperature:g} K endpoints")
    return sorted(found)


def mean_band(frames, key, value):
    """Mean and sample SD of one column across replicas.

    Parameters
    ----------
    frames : list of pandas.DataFrame
        One table per replica.
    key : str or list of str
        Columns that identify a point.
    value : str
        Column to average.

    Returns
    -------
    pandas.DataFrame
        ``key`` columns, ``mean``, ``sd`` and ``n``.
    """
    stacked = pd.concat(frames, ignore_index=True)
    grouped = stacked.groupby(key)[value]
    return grouped.agg(mean="mean", sd="std", n="count").reset_index()


def hold_temperatures(results):
    """Temperatures of the hold rows.

    Parameters
    ----------
    results : pathlib.Path
        The ``results`` directory.

    Returns
    -------
    set of float
        Hold temperatures in kelvin.
    """
    return {float(row["temperature_K"]) for row in manifest_rows(results) if row["kind"] == "hold"}


# ---------------------------------------------------------------------------
# Figure 1
# ---------------------------------------------------------------------------

@panel("Fig1b_cooling_protocols")
def fig1b(results, out, stem):
    """Thermostat temperature against time for each cooling rate."""
    files = sorted(Path(results, "energy").glob("ramp_*_r1.csv"))
    if not files:
        raise FileNotFoundError("no results/energy/ramp_<rate>_r1.csv")
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    for path in sorted(files, key=lambda p: float(p.stem.split("_")[1])):
        rate = float(path.stem.split("_")[1])
        frame = pd.read_csv(path)
        time_ns = (frame["time_ps"] - frame["time_ps"].min()) / 1000.0
        axis.plot(time_ns, frame["temp_K"], lw=1.0, color=RATE_COLOR_BY_VALUE.get(rate),
                  label=f"{rate:g} K/ns")
    axis.set(xlabel="Time (ns)", ylabel="Thermostat temperature (K)")
    axis.grid(alpha=0.2)
    axis.legend(title="Cooling rate")
    return save(figure, out, stem)


# ---------------------------------------------------------------------------
# Figure 4 and structure SI
# ---------------------------------------------------------------------------

def _density_dipole(results, out, stem, temperature):
    """Density and dipole orientation profiles of one endpoint temperature."""
    frames = [pd.read_csv(need(directory / "dipole" / "water_orientation_z_profile.csv")).assign(replica=rep)
              for rep, directory in endpoint_dirs(results, temperature)]
    rho = mean_band(frames, "z", "rho_g_cm3")
    cos = mean_band(frames, "z", "mean_cos_theta")
    figure, axis = plt.subplots(figsize=(4.6, 5.6))
    axis.plot(rho["mean"], rho["z"], color="#1f3b8c", lw=1.6, label=r"$\rho(z)$")
    axis.fill_betweenx(rho["z"], rho["mean"] - rho["sd"], rho["mean"] + rho["sd"], color="#1f3b8c", alpha=0.2, lw=0)
    bulk = rho.loc[rho["z"].abs() <= BULK_HALFWIDTH_A, "mean"].mean()
    above = rho["z"][(rho["mean"] >= 0.5 * bulk)]
    for z in (above.min(), above.max()):
        axis.axhline(z, color="0.3", ls="--", lw=0.75)
    axis.set(xlabel=r"Oxygen-derived water density (g cm$^{-3}$)", ylabel=r"$z$ from slab center (Å)",
             title=f"{temperature:g} K")
    axis.set_xlim(0, 1.25)
    top = axis.twiny()
    shown = cos[cos["z"].between(above.min() - 15, above.max() + 15)]
    top.plot(shown["mean"], shown["z"], color="#c0392b", lw=1.3, label=r"$\langle\cos\theta\rangle$")
    top.set_xlabel(r"$\langle\cos\theta\rangle$ of the dipole to $z$", color="#c0392b")
    top.tick_params(axis="x", colors="#c0392b")
    top.set_xlim(-0.5, 0.5)
    handles = axis.get_lines()[:1] + top.get_lines()
    axis.legend(handles, [h.get_label() for h in handles])
    return save(figure, out, stem)


@panel("Fig4a_density_dipole_300K")
def fig4a(results, out, stem):
    """Density and dipole profile at 300 K."""
    return _density_dipole(results, out, stem, 300.0)


@panel("Fig4b_density_dipole_78K")
def fig4b(results, out, stem):
    """Density and dipole profile at 78 K."""
    return _density_dipole(results, out, stem, 78.0)


def _sq_depth(results, temperature, depth_max=60.0):
    """Replica-mean depth-resolved S(q) at one endpoint temperature."""
    frames = []
    for rep, directory in endpoint_dirs(results, temperature):
        table = pd.read_csv(need(directory / "local_order" / "water_Sq2D_by_depth.csv"))
        table["depth_key"] = np.where(table["is_gibbs_surface"].astype(str).str.lower() == "true",
                                      -1.0, table["depth_A"].round(1))
        frames.append(table.assign(replica=rep))
    stacked = pd.concat(frames, ignore_index=True)
    stacked = stacked[(stacked["depth_key"] < 0) | (stacked["depth_key"] <= depth_max)]
    stacked["q_key"] = stacked["q_Ainv"].round(4)
    curves = stacked.groupby(["depth_key", "q_key"]).agg(
        depth_A=("depth_A", "mean"), Sq=("Sq_mean", "mean"), sd=("Sq_mean", "std")).reset_index()
    return curves


def _sq_depth_panel(results, out, stem, temperature, depth_max, bands):
    """Depth-resolved S(q) colored by distance from the center."""
    curves = _sq_depth(results, temperature, depth_max)
    norm = matplotlib.colors.Normalize(0, depth_max)
    cmap = matplotlib.colormaps["viridis"]
    figure, axis = plt.subplots(figsize=(6.0, 4.3))
    for key, curve in curves.groupby("depth_key"):
        curve = curve.sort_values("q_key")
        if key < 0:
            depth = curve["depth_A"].mean()
            axis.plot(curve["q_key"], curve["Sq"], color=cmap(norm(min(depth, depth_max))), ls="--", lw=2.0,
                      label=f"Half-density surface, |z| = {depth:.0f} Å")
            continue
        center = abs(key) < 1e-6
        axis.plot(curve["q_key"], curve["Sq"], color=cmap(norm(key)), lw=2.2 if center else 0.9,
                  label="Center, |z| = 0" if center else "_nolegend_")
        if bands and center:
            axis.fill_between(curve["q_key"], curve["Sq"] - curve["sd"], curve["Sq"] + curve["sd"],
                              color=cmap(norm(key)), alpha=0.2, lw=0)
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    bar = figure.colorbar(mappable, ax=axis, pad=0.02)
    bar.set_label("Distance from the slab center |z| (Å)")
    axis.set(xlabel=r"Wave vector $q$ (Å$^{-1}$)", ylabel=r"In-plane oxygen $S^{\parallel}(q)$",
             title=f"{temperature:g} K", xlim=(0.5, 5.0))
    axis.grid(alpha=0.2)
    axis.legend()
    return save(figure, out, stem)


@panel("Fig4c_Sq_depth_300K")
def fig4c(results, out, stem):
    """Depth-resolved in-plane S(q) at 300 K."""
    return _sq_depth_panel(results, out, stem, 300.0, 60.0, True)


@panel("Fig4d_Sq_depth_78K")
def fig4d(results, out, stem):
    """Depth-resolved in-plane S(q) at 78 K."""
    return _sq_depth_panel(results, out, stem, 78.0, 60.0, True)


def _sq_map(results, out, stem, temperature):
    """In-plane S(qx, qy) of the film center, replica mean."""
    maps, qx, qy = [], None, None
    for _rep, directory in endpoint_dirs(results, temperature):
        data = np.load(need(directory / "local_order" / "water_Sq2D_center_bragg_map.npz"))
        image = np.asarray(data["Sq_mean"], dtype=float)
        # Put qy on the rows and qx on the columns, as imshow expects.
        if "QX" in data and np.ptp(data["QX"][:, 0]) > np.ptp(data["QX"][0, :]):
            image = image.T
        if qx is None:
            qx, qy = np.asarray(data["qx"]).ravel(), np.asarray(data["qy"]).ravel()
        if maps and image.shape != maps[0].shape:
            continue  # a replica on a different grid is left out of the mean
        maps.append(image)
    image = np.nanmean(maps, axis=0)
    order_x, order_y = np.argsort(qx), np.argsort(qy)
    image = image[np.ix_(order_y, order_x)]
    qx, qy = qx[order_x], qy[order_y]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_image = np.log10(np.where(image > 0, image, np.nan))
    figure, axis = plt.subplots(figsize=(5.2, 4.6))
    shown = axis.imshow(log_image, origin="lower", cmap="viridis", vmin=-0.7, vmax=0.7,
                        extent=[qx.min(), qx.max(), qy.min(), qy.max()], aspect="equal")
    axis.set(xlim=(-5, 5), ylim=(-5, 5), xlabel=r"$q_x$ (Å$^{-1}$)", ylabel=r"$q_y$ (Å$^{-1}$)",
             title=f"{temperature:g} K, film center")
    bar = figure.colorbar(shown, ax=axis, pad=0.02)
    bar.set_label(r"$\log_{10} S^{\parallel}(q_x, q_y)$")
    return save(figure, out, stem)


@panel("Fig4e_Sq_map_300K")
def fig4e(results, out, stem):
    """In-plane reciprocal-space map at 300 K."""
    return _sq_map(results, out, stem, 300.0)


@panel("Fig4f_Sq_map_78K")
def fig4f(results, out, stem):
    """In-plane reciprocal-space map at 78 K."""
    return _sq_map(results, out, stem, 78.0)


def _bulk_landscape(results, temperature):
    """Per-water q, Sk and LSI of bulk water, all replicas pooled."""
    frames = []
    for rep, directory in endpoint_dirs(results, temperature):
        base = directory / "local_order"
        keys = ["frame", "oxygen_index"]
        q = pd.read_csv(need(base / "q_per_water_distribution.csv"), usecols=keys + ["z_coord", "q"])
        sk = pd.read_csv(need(base / "Sk_per_water_distribution.csv"), usecols=keys + ["Sk"])
        lsi = pd.read_csv(need(base / "LSI_per_water_distribution.csv"), usecols=keys + ["LSI"])
        joined = q.merge(sk, on=keys).merge(lsi, on=keys)
        frames.append(joined[joined["z_coord"].abs() <= BULK_HALFWIDTH_A].assign(replica=rep))
    return pd.concat(frames, ignore_index=True)


def _landscape(results, out, stem, temperature, sk_range):
    """Population of (q, Sk) bins as bars colored by the median LSI."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, registers the 3D projection

    data = _bulk_landscape(results, temperature)
    q_edges = np.linspace(0.5, 1.0, 26)
    sk_edges = np.linspace(sk_range[0], sk_range[1], 26)
    counts, _, _ = np.histogram2d(data["q"], data["Sk"], bins=[q_edges, sk_edges])
    fraction = counts / max(len(data), 1)
    q_index = np.digitize(data["q"], q_edges) - 1
    sk_index = np.digitize(data["Sk"], sk_edges) - 1
    keep = (q_index >= 0) & (q_index < len(q_edges) - 1) & (sk_index >= 0) & (sk_index < len(sk_edges) - 1)
    median = pd.DataFrame({"i": q_index[keep], "j": sk_index[keep], "LSI": data["LSI"].to_numpy()[keep]}) \
        .groupby(["i", "j"])["LSI"].median()
    norm = matplotlib.colors.Normalize(0.0, 0.2)
    cmap = matplotlib.colormaps["viridis"]
    figure = plt.figure(figsize=(6.4, 5.4))
    axis = figure.add_subplot(projection="3d")
    dq, dsk = np.diff(q_edges)[0], np.diff(sk_edges)[0]
    for (i, j), value in median.items():
        if fraction[i, j] <= 0:
            continue
        axis.bar3d(q_edges[i], sk_edges[j], 0.0, dq * 0.9, dsk * 0.9, fraction[i, j],
                   color=cmap(norm(value)), shade=True, linewidth=0)
    axis.set_xlabel(r"Orientational tetrahedrality $q_{tet}$")
    axis.set_ylabel(r"Translational tetrahedrality $S_k$")
    axis.set_zlabel("Population fraction")
    axis.set_title(f"Bulk water, {temperature:g} K")
    axis.set_xlim(q_edges[0], q_edges[-1])
    axis.set_ylim(sk_edges[0], sk_edges[-1])
    bar = figure.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axis, shrink=0.6, pad=0.12)
    bar.set_label(r"Median LSI (Å$^2$)")
    return save(figure, out, stem)


def _shared_sk_range(results):
    """Common Sk axis for the two landscapes, from the pooled 1st percentile."""
    values = pd.concat([_bulk_landscape(results, t)["Sk"] for t in (300.0, 78.0)])
    low = float(np.nanpercentile(values, 1.0))
    return (min(low, 0.997), 1.0)


@panel("Fig5a_landscape_300K")
def fig5a(results, out, stem):
    """Bulk q, Sk and LSI landscape at 300 K."""
    return _landscape(results, out, stem, 300.0, _shared_sk_range(results))


@panel("Fig5b_landscape_78K")
def fig5b(results, out, stem):
    """Bulk q, Sk and LSI landscape at 78 K."""
    return _landscape(results, out, stem, 78.0, _shared_sk_range(results))


@panel("Fig5c_cross_sections")
def fig5c(results, out, stem):
    """Grid of the colored hold cross sections: replicas in rows, 250 to 150 K in columns."""
    folder = Path(results, "lda_frames", "phase_density_maps")
    images = {}
    for path in folder.glob("T*_r*_liquid_LDA_sharp.png"):
        match = re.match(r"T(\d+)_r(\w+?)_liquid_LDA_sharp", path.stem)
        if match:
            images[(int(match.group(1)), match.group(2))] = path
    if not images:
        raise FileNotFoundError(f"no colored cross sections in {folder}")
    temperatures = sorted({t for t, _ in images}, reverse=True)
    replicas = sorted({r for _, r in images})
    figure, axes = plt.subplots(len(replicas), len(temperatures), squeeze=False,
                                figsize=(1.3 * len(temperatures), 1.1 * len(replicas) + 0.4))
    for i, replica in enumerate(replicas):
        for j, temperature in enumerate(temperatures):
            axis = axes[i, j]
            axis.set_xticks([])
            axis.set_yticks([])
            if (temperature, replica) in images:
                axis.imshow(plt.imread(images[(temperature, replica)]))
            if i == 0:
                axis.set_title(f"{temperature} K", fontsize=8)
            if j == 0:
                axis.set_ylabel(f"r{replica}", fontsize=8)
    figure.suptitle("Red: liquid-like. Blue: LDA-like (display classifier, 250 K vs 150 K).", fontsize=8)
    return save(figure, out, stem)


def _classifier_regions(results, column, ylabel, out, stem):
    """Hold value of one classifier column for bulk and AWI."""
    table = pd.read_csv(need(Path(results, "classifier", "region_summary.tsv")), sep="\t")
    table = table[table["kind"] == "hold"]
    figure, axis = plt.subplots(figsize=(5.6, 4.2))
    for region, label in ((BULK, "Bulk"), (CLASSIFIER_AWI, "AWI")):
        part = table[table["region"] == region].sort_values("temperature_K")
        if part.empty:
            continue
        axis.plot(part["temperature_K"], part[f"{column}_mean"], marker="o", ms=4, label=label)
        axis.fill_between(part["temperature_K"], part[f"{column}_mean"] - part[f"{column}_sd"],
                          part[f"{column}_mean"] + part[f"{column}_sd"], alpha=0.2, lw=0)
    axis.set(xlabel="Temperature (K)", ylabel=ylabel)
    axis.grid(alpha=0.2)
    axis.legend()
    return save(figure, out, stem)


@panel("Fig5d_LDA_fraction")
def fig5d(results, out, stem):
    """Fraction of waters classified as LDA-like on the holds."""
    return _classifier_regions(results, "f_lda", "Fraction of waters classified LDA-like", out, stem)


@panel("Fig5e_free_volume")
def fig5e(results, out, stem):
    """Probe-accessible free volume on the holds."""
    return _classifier_regions(results, "free_volume_fraction", "Probe-accessible free-volume fraction", out, stem)


# ---------------------------------------------------------------------------
# Figures 6 and 7
# ---------------------------------------------------------------------------

def _regional_curves(table, value, ylabel, out, stem, lag_label="Lag time (ps)", threshold=True):
    """Replica-mean curves of bulk and AWI at the three comparison temperatures."""
    table = table[np.isclose(table["rate"].astype(float), RATE)]
    figure, axis = plt.subplots(figsize=(6.0, 4.2))
    for temperature, color in TEMPERATURE_COLORS.items():
        for region, label in ((BULK, "Bulk"), (AWI, "AWI")):
            part = table[np.isclose(table["temp_mid_K"], temperature) & (table["region"] == region)]
            if part.empty:
                continue
            band = part.groupby("lag_ps")[value].agg(["mean", "std"]).reset_index()
            axis.plot(band["lag_ps"], band["mean"], color=color, label=f"{temperature:g} K, {label}")
            axis.fill_between(band["lag_ps"], band["mean"] - band["std"], band["mean"] + band["std"],
                              color=color, alpha=0.12, lw=0)
    if threshold:
        axis.axhline(np.exp(-1.0), color="0.4", ls=":", lw=0.75)
    axis.set(xscale="log", xlabel=lag_label, ylabel=ylabel)
    axis.grid(alpha=0.2)
    axis.legend()
    return save(figure, out, stem)


def _replica_curves(results):
    return pd.read_csv(need(Path(results, "figures", "work", "regional_dynamics", "replica_curves.csv")))


@panel("Fig6a_Fs_bulk_AWI")
def fig6a(results, out, stem):
    """In-plane F_s of bulk and AWI."""
    return _regional_curves(_replica_curves(results), "Fs", r"$F_s^{\parallel}(q^*, t)$", out, stem)


@panel("Fig6b_C1_bulk_AWI")
def fig6b(results, out, stem):
    """C1 of bulk and AWI."""
    return _regional_curves(_replica_curves(results), "C1", r"$C_1(t)$", out, stem)


@panel("Fig6c_C2_bulk_AWI")
def fig6c(results, out, stem):
    """C2 of bulk and AWI."""
    return _regional_curves(_replica_curves(results), "C2", r"$C_2(t)$", out, stem)


def _hb_table(results):
    return pd.read_csv(need(Path(results, "figures", "work", "hbond", "hb_intermittent_replica.csv")))


@panel("Fig7a_HB_retention")
def fig7a(results, out, stem):
    """Intermittent H-bond partner retention of bulk and AWI."""
    return _regional_curves(_hb_table(results), "C_HB_intermittent",
                            "Intermittent original-partner H-bond retention", out, stem, threshold=False)


@panel("Fig7b_HB_retention_contrast_320ps")
def fig7b(results, out, stem):
    """Paired bulk - AWI retention at 320 ps along the 4.44 K/ns ramp."""
    table = _hb_table(results)
    table = table[np.isclose(table["rate"].astype(float), RATE) & np.isclose(table["lag_ps"], 320.0)]
    table = table[~table["temp_mid_K"].isin(hold_temperatures(results))]
    wide = table.pivot_table(index=["replica", "temp_mid_K"], columns="region", values="C_HB_intermittent")
    wide = wide.dropna(subset=[BULK, AWI])
    rows = []
    for temperature, group in (wide[BULK] - wide[AWI]).groupby(level="temp_mid_K"):
        mean, _sd, half, count = paired_student_interval(group.to_numpy())
        rows.append((temperature, mean, half, count))
    summary = pd.DataFrame(rows, columns=["T", "mean", "half", "n"]).sort_values("T")
    summary.to_csv(out / f"{stem}.csv", index=False)
    figure, axis = plt.subplots(figsize=(5.6, 4.2))
    axis.plot(summary["T"], summary["mean"], marker="o", ms=4, label="Bulk − AWI, 320 ps")
    axis.fill_between(summary["T"], summary["mean"] - summary["half"], summary["mean"] + summary["half"],
                      alpha=0.3, lw=0)
    axis.axhline(0, color="0.4", ls=":", lw=0.75)
    axis.set(xlabel="Cooling-window midpoint temperature (K)", ylabel="(Bulk − AWI) H-bond retention",
             title="Band: paired Student-t 95% interval")
    axis.grid(alpha=0.2)
    axis.legend()
    return save(figure, out, stem)


def _crossing_time(lags, values, level=np.exp(-1.0)):
    """Lag at which a decreasing curve first reaches ``level``, log-interpolated."""
    lags, values = np.asarray(lags, float), np.asarray(values, float)
    below = np.flatnonzero(values <= level)
    if not below.size or below[0] == 0:
        return np.nan
    i = below[0]
    x0, x1 = np.log(lags[i - 1]), np.log(lags[i])
    y0, y1 = values[i - 1], values[i]
    return float(np.exp(x0 + (level - y0) * (x1 - x0) / (y1 - y0)))


@panel("Fig7e_tau_vs_HB_time")
def fig7e(results, out, stem):
    """Resolved tau_alpha against the e^-1 time of H-bond retention."""
    tau = pd.read_csv(need(Path(results, "figures", "work", "regional_dynamics", "tau_censoring_corrected.csv")))
    tau = tau[(tau["metric"] == "Fs") & (tau["status"] == "resolved") & np.isclose(tau["rate"].astype(float), RATE)]
    hb = _hb_table(results)
    hb = hb[np.isclose(hb["rate"].astype(float), RATE)]
    times = []
    for (replica, temperature, region), group in hb.groupby(["replica", "temp_mid_K", "region"]):
        group = group.sort_values("lag_ps")
        times.append((replica, temperature, region, _crossing_time(group["lag_ps"], group["C_HB_intermittent"])))
    hb_time = pd.DataFrame(times, columns=["replica", "temperature_K", "region", "tau_hb_ps"])
    pairs = tau.merge(hb_time, on=["replica", "temperature_K", "region"]).dropna(subset=["tau_ps", "tau_hb_ps"])
    pairs.to_csv(out / f"{stem}.csv", index=False)
    if pairs.empty:
        raise ValueError("no temperature has both a resolved tau and an H-bond retention time")
    figure, axis = plt.subplots(figsize=(5.4, 4.4))
    for region, label, marker in ((BULK, "Bulk", "o"), (AWI, "AWI", "s")):
        part = pairs[pairs["region"] == region]
        if part.empty:
            continue
        mean = part.groupby("temperature_K")[["tau_hb_ps", "tau_ps"]].mean()
        x, y = np.log10(mean["tau_hb_ps"]), np.log10(mean["tau_ps"])
        axis.plot(mean["tau_hb_ps"], mean["tau_ps"], marker=marker, ls="none", color="black",
                  markerfacecolor="black" if region == BULK else "none", label=label)
        if len(mean) >= 2:
            slope, intercept = np.polyfit(x, y, 1)
            grid = np.linspace(x.min(), x.max(), 50)
            axis.plot(10 ** grid, 10 ** (intercept + slope * grid), lw=1.0, color="black",
                      label=f"{label} fit, slope {slope:.2f}")
    axis.set(xscale="log", yscale="log", xlabel=r"Intermittent H-bond retention $e^{-1}$ time (ps)",
             ylabel=r"$\tau_{\alpha,\parallel}$ (ps)")
    axis.grid(alpha=0.2)
    axis.legend()
    return save(figure, out, stem)


# ---------------------------------------------------------------------------
# SI: endpoint structure
# ---------------------------------------------------------------------------

def _profiles(results, temperature, name):
    frames = [pd.read_csv(need(directory / "local_order" / name)).assign(replica=rep)
              for rep, directory in endpoint_dirs(results, temperature)]
    return pd.concat(frames, ignore_index=True)


@panel("SI_order_profiles")
def si_order_profiles(results, out, stem):
    """q, Sk and LSI against z at 300 and 78 K, with the 78 - 300 K change."""
    columns = (("mean_q_orientational_tetrahedral", r"$\langle q \rangle$"),
               ("mean_Sk_translational_tetrahedral", r"$\langle S_k \rangle$"),
               ("mean_LSI", r"$\langle$LSI$\rangle$ (Å$^2$)"))
    tables = {t: _profiles(results, t, "water_local_order_z_profile_10A.csv") for t in (300.0, 78.0)}
    written = []
    for column, label in columns:
        figure, axis = plt.subplots(figsize=(5.4, 4.0))
        for temperature in (300.0, 78.0):
            band = tables[temperature].groupby("z_center")[column].agg(["mean", "std"]).reset_index()
            band = band[band["mean"].notna()]
            axis.plot(band["z_center"], band["mean"], marker="o", ms=3, label=f"{temperature:g} K")
            axis.fill_between(band["z_center"], band["mean"] - band["std"], band["mean"] + band["std"], alpha=0.2, lw=0)
        axis.set(xlabel=r"$z$ from slab center (Å)", ylabel=label)
        axis.grid(alpha=0.2)
        axis.legend()
        short = column.split("_")[1]
        written += save(figure, out, f"{stem}_{short}")
        cold = tables[78.0].groupby("z_center")[column].mean()
        warm = tables[300.0].groupby("z_center")[column].mean()
        delta = (cold - warm).dropna()
        figure, axis = plt.subplots(figsize=(5.4, 4.0))
        axis.plot(delta.index, delta.values, marker="o", ms=3, label="78 K − 300 K")
        axis.axhline(0, color="0.4", ls=":", lw=0.75)
        axis.set(xlabel=r"$z$ from slab center (Å)", ylabel=f"Δ{label}")
        axis.grid(alpha=0.2)
        axis.legend()
        written += save(figure, out, f"{stem}_delta_{short}")
    return written


@panel("SI_ptheta")
def si_ptheta(results, out, stem):
    """O-O-O angle distribution in the center and next to the interface."""
    written = []
    tables = {t: _profiles(results, t, "water_water_angular_distribution_Ptheta_by_z_window.csv") for t in (300.0, 78.0)}
    windows = [c for c in tables[78.0].columns if c.startswith("z_")]
    bounds = {c: tuple(float(v) for v in re.findall(r"-?\d+", c)) for c in windows}
    center = [c for c in windows if abs(0.5 * sum(bounds[c])) <= BULK_HALFWIDTH_A]
    for name, chosen in (("center", center),):
        figure, axis = plt.subplots(figsize=(5.4, 4.0))
        for temperature in (300.0, 78.0):
            table = tables[temperature]
            curve = table.groupby("theta_deg")[chosen].mean().mean(axis=1)
            axis.plot(curve.index, curve.values, label=f"{temperature:g} K")
        axis.set(xlabel=r"O–O–O angle $\theta$ (degrees)", ylabel=r"$P(\theta)$ (deg$^{-1}$)",
                 title=f"Film center, {len(chosen)} windows within |z| ≤ 20 Å")
        axis.grid(alpha=0.2)
        axis.legend()
        written += save(figure, out, f"{stem}_{name}")
    for temperature in (300.0, 78.0):
        table = tables[temperature]
        populated = [c for c in windows if table[c].notna().any()]
        figure, axis = plt.subplots(figsize=(6.0, 4.2))
        cmap = matplotlib.colormaps["viridis"]
        depth = {c: abs(0.5 * sum(bounds[c])) for c in populated}
        top = max(depth.values()) if depth else 1.0
        for column in sorted(populated, key=lambda c: depth[c]):
            curve = table.groupby("theta_deg")[column].mean()
            axis.plot(curve.index, curve.values, color=cmap(depth[column] / top), lw=0.9, label=column.replace("_", " "))
        axis.set(xlabel=r"O–O–O angle $\theta$ (degrees)", ylabel=r"$P(\theta)$ (deg$^{-1}$)", title=f"{temperature:g} K")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=6, ncol=2)
        written += save(figure, out, f"{stem}_{temperature:g}K_by_depth")
    return written


@panel("SI_chill")
def si_chill(results, out, stem):
    """CHILL+ classes against z at 300 and 78 K."""
    written = []
    for temperature in (300.0, 78.0):
        table = _profiles(results, temperature, "CHILLplus_z_profile_10A.csv")
        figure, axis = plt.subplots(figsize=(6.0, 4.2))
        for label, name in CHILL_TYPES.items():
            column = f"fraction_type_{label}"
            if column not in table:
                continue
            band = table.groupby("z_center")[column].agg(["mean", "std"]).reset_index().dropna(subset=["mean"])
            axis.plot(band["z_center"], band["mean"], marker="o", ms=3, label=name)
        axis.set(xlabel=r"$z$ from slab center (Å)", ylabel="CHILL+ fraction", title=f"{temperature:g} K")
        axis.set_yscale("symlog", linthresh=1e-3)
        axis.grid(alpha=0.2)
        axis.legend()
        written += save(figure, out, f"{stem}_{temperature:g}K")
    return written


@panel("SI_sq_depth")
def si_sq_depth(results, out, stem):
    """Depth-resolved S(q) out to the vapor side of the interface."""
    written = []
    for temperature in (300.0, 78.0):
        written += _sq_depth_panel(results, out, f"{stem}_{temperature:g}K", temperature, 70.0, False)
    return written


# ---------------------------------------------------------------------------
# SI: holds and ramps
# ---------------------------------------------------------------------------

@panel("SI_fv_radius")
def si_fv_radius(results, out, stem):
    """Free-volume fraction against temperature for each probe radius."""
    table = pd.read_csv(need(Path(results, "classifier", "frame_region_metrics.tsv")), sep="\t")
    table = table[table["kind"] == "hold"]
    radii = {CONFIG.probe.radius_A: "free_volume_fraction"}
    for column in table.columns:
        match = re.fullmatch(r"free_volume_r([0-9.]+)", column)
        if match:
            radii[float(match.group(1))] = column
    written = []
    for region, label in ((BULK, "Bulk"), (CLASSIFIER_AWI, "AWI")):
        part = table[table["region"] == region]
        if part.empty:
            raise ValueError(f"no {region} rows in frame_region_metrics.tsv")
        figure, axis = plt.subplots(figsize=(5.8, 4.2))
        cmap = matplotlib.colormaps["viridis"]
        top = max(radii) or 1.0
        for radius in sorted(radii):
            column = radii[radius]
            per_replica = part.groupby(["temperature_K", "replica"])[column].mean()
            band = per_replica.groupby(level="temperature_K").agg(["mean", "std"]).reset_index()
            axis.plot(band["temperature_K"], band["mean"], marker="o", ms=3, color=cmap(radius / top * 0.9),
                      label=f"r = {radius:g} Å")
            axis.fill_between(band["temperature_K"], band["mean"] - band["std"], band["mean"] + band["std"],
                              color=cmap(radius / top * 0.9), alpha=0.18, lw=0)
        axis.set(xlabel="Temperature (K)", ylabel="Probe-accessible free-volume fraction", title=label, yscale="log")
        axis.grid(alpha=0.2)
        axis.legend(title="Probe radius")
        written += save(figure, out, f"{stem}_{label.lower()}")
    return written


@panel("SI_conservation")
def si_conservation(results, out, stem):
    """Largest relative mass error of each ramp window, by cooling rate."""
    rows = []
    for row in manifest_rows(results):
        if row["kind"] != "ramp":
            continue
        path = result_directory(results.parent, "profiles", row) / "frame_geometry.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path, usecols=["mass_relative_error", "areal_mass"])
        rows.append(dict(rate=float(row["rate_K_per_ns"]), replica=row["replica"],
                         T=float(row["temperature_K"]),
                         max_error=float(frame["mass_relative_error"].abs().max()),
                         areal_cv=float(frame["areal_mass"].std() / frame["areal_mass"].mean())))
    if not rows:
        raise FileNotFoundError("no results/profiles/<row>/frame_geometry.csv")
    table = pd.DataFrame(rows)
    table.to_csv(out / f"{stem}.csv", index=False)
    figure, axis = plt.subplots(figsize=(6.0, 4.2))
    for rate, part in table.groupby("rate"):
        worst = part.groupby("T")["max_error"].max()
        axis.plot(worst.index, worst.values, marker="o", ms=3, color=RATE_COLOR_BY_VALUE.get(rate), label=f"{rate:g} K/ns")
    axis.set(xlabel="Cooling-window midpoint temperature (K)", ylabel="Largest |relative mass error| per window",
             yscale="log")
    axis.grid(alpha=0.2)
    axis.legend(title="Cooling rate")
    return save(figure, out, stem)


def read_cpptraj_table(path):
    """Read a cpptraj data file with a ``#`` header line.

    Parameters
    ----------
    path : pathlib.Path
        cpptraj output.

    Returns
    -------
    pandas.DataFrame
        Numeric columns named from the last header line, or ``c0, c1, ...``.
    """
    header, rows = None, []
    for line in Path(path).read_text().splitlines():
        text = line.strip()
        if not text:
            continue
        if text.startswith("#"):
            header = text.lstrip("#").split()
            continue
        try:
            rows.append([float(value) for value in text.split()])
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"no numeric rows in {path}")
    width = max(len(r) for r in rows)
    rows = [r + [np.nan] * (width - len(r)) for r in rows]
    names = header if header and len(header) == width else [f"c{i}" for i in range(width)]
    return pd.DataFrame(rows, columns=names)


def _hold_capillary(results, name):
    """cpptraj capillary tables of the r1 holds, by temperature."""
    found = []
    for row in manifest_rows(results):
        if row["kind"] != "hold" or row["replica"] != "r1":
            continue
        path = result_directory(results.parent, "capillary", row) / name
        if path.is_file():
            found.append((float(row["temperature_K"]), read_cpptraj_table(path)))
    if not found:
        raise FileNotFoundError(f"no results/capillary/<row>/{name}")
    return sorted(found)


@panel("SI_roughness")
def si_roughness(results, out, stem):
    """Interface roughness against time on each r1 hold (first column against the second)."""
    tables = _hold_capillary(results, "capwave_roughness.dat")
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    cmap = matplotlib.colormaps["plasma"]
    temperatures = [t for t, _ in tables]
    norm = matplotlib.colors.Normalize(min(temperatures), max(temperatures) + 0.2 * (max(temperatures) - min(temperatures) + 1))
    for temperature, table in tables:
        x, y = table.columns[0], table.columns[1]
        axis.plot(table[x], table[y], lw=0.75, color=cmap(norm(temperature)), label=f"{temperature:g} K")
    axis.set(xlabel=tables[0][1].columns[0], ylabel=tables[0][1].columns[1], title="Replica r1 holds")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7, ncol=2)
    return save(figure, out, stem)


@panel("SI_capillary")
def si_capillary(results, out, stem):
    """Capillary-wave spectrum of each r1 hold (first column against the second, log-log)."""
    tables = _hold_capillary(results, "capwave_spectrum.dat")
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    cmap = matplotlib.colormaps["plasma"]
    temperatures = [t for t, _ in tables]
    norm = matplotlib.colors.Normalize(min(temperatures), max(temperatures) + 0.2 * (max(temperatures) - min(temperatures) + 1))
    for temperature, table in tables:
        x, y = table.columns[0], table.columns[1]
        part = table[(table[x] > 0) & (table[y] > 0)]
        axis.plot(part[x], part[y], marker="o", ms=2, lw=0.75, color=cmap(norm(temperature)), label=f"{temperature:g} K")
    axis.set(xscale="log", yscale="log", xlabel=tables[0][1].columns[0], ylabel=tables[0][1].columns[1],
             title="Replica r1 holds")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7, ncol=2)
    return save(figure, out, stem)


def main(argv=None):
    """Draw the requested panels.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments, excluding the program name.

    Returns
    -------
    int
        Zero. Panels that could not be drawn are listed in
        ``FAILED_panels.txt``.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results/figures/panels"))
    parser.add_argument("--only", nargs="*", default=None, help="Panel names or prefixes to draw.")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    failed = []
    for name, function in PANELS.items():
        if args.only and not any(name.startswith(prefix) for prefix in args.only):
            continue
        try:
            files = function(args.results.resolve(), args.out, name)
            print(f"OK   {name}: {len(files)} files")
        except Exception as error:  # one panel must not stop the others
            failed.append(f"{name}: {type(error).__name__}: {error}")
            print(f"SKIP {name}: {type(error).__name__}: {error}")
            if not isinstance(error, FileNotFoundError):
                traceback.print_exc()
    report = args.out / "FAILED_panels.txt"
    if failed:
        report.write_text("\n".join(failed) + "\n")
    elif report.exists():
        report.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
