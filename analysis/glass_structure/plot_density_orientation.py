#!/usr/bin/env python3
"""Mass density and dipole orientation across the film.

The input is the cpptraj ``density ... number`` profile. That column is
already a number density in Å^-3, so the mass density is
``n * M / N_A * 10^24`` with no extra division by area or bin width.
The bulk density of a 300 K frame must be about 1 g cm^-3. Displayed
curves and the half-density surface use the same Gaussian of width 1 Å.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from opcfilm.plotting.conventions import install_figure_conventions
install_figure_conventions()  # temperature axes hot to cold; legends outside panels
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.constants import (
    BULK_HALFWIDTH_A,
    NUMBER_TO_MASS_DENSITY,
    PROFILE_SMOOTH_SIGMA_A,
    DEFAULT_LX_A,
    DEFAULT_LY_A,
)
from common.slab import half_density_surfaces


def smooth_profile(y, z, sigma_A=PROFILE_SMOOTH_SIGMA_A):
    """Gaussian-smooth a profile with a fixed length in angstroms.

    Parameters
    ----------
    y : array_like
        Profile values on ``z``.
    z : array_like
        Bin centers in angstroms.
    sigma_A : float, optional
        Gaussian standard deviation in angstroms.

    Returns
    -------
    numpy.ndarray
        Smoothed profile, in the original order of ``y``.
    """
    y = np.asarray(y, dtype=float)
    order = np.argsort(np.asarray(z, dtype=float))
    spacing = float(np.median(np.diff(np.asarray(z, dtype=float)[order])))
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("Profile z coordinates must increase.")
    smoothed = np.empty_like(y)
    smoothed[order] = gaussian_filter1d(y[order], sigma=sigma_A / spacing, mode="nearest")
    return smoothed


def number_density_to_mass(number_density_A3):
    """Convert a cpptraj number density to mass density.

    Parameters
    ----------
    number_density_A3 : array_like
        Number density in molecules per cubic angstrom, as written by
        ``density ... number``. This is not a count per slab.

    Returns
    -------
    numpy.ndarray
        Mass density in g cm^-3.
    """
    return np.asarray(number_density_A3, dtype=float) * NUMBER_TO_MASS_DENSITY


def load_density_file(filename):
    """Read one cpptraj oxygen number-density profile.

    Parameters
    ----------
    filename : path-like
        Whitespace-separated file. The first column is z and the second
        is the number density in Å^-3.

    Returns
    -------
    pandas.DataFrame
        Columns ``z``, ``density_g_cm3``, and ``number_density_A3``.
    """
    frame = pd.read_csv(
        filename,
        comment="#",
        sep=r"\s+",
        header=None,
    )
    if frame.shape[1] < 2:
        raise ValueError(f"{filename} needs z and a number-density column.")
    out = pd.DataFrame({
        "z": frame.iloc[:, 0].to_numpy(float),
        "number_density_A3": frame.iloc[:, 1].to_numpy(float),
    })
    out["density_g_cm3"] = number_density_to_mass(out["number_density_A3"])
    return out[["z", "density_g_cm3", "number_density_A3"]]


def load_cos_file(filename):
    """Read a dipole-orientation profile.

    Parameters
    ----------
    filename : path-like
        CSV with columns ``z`` and ``mean_cos_theta``.

    Returns
    -------
    pandas.DataFrame
        Those two columns.
    """
    frame = pd.read_csv(filename)
    return frame[["z", "mean_cos_theta"]]


def _mean_sd(frame, columns):
    """Mean and sample standard deviation across replica columns.

    Parameters
    ----------
    frame : pandas.DataFrame
        Merged replica table.
    columns : sequence of str
        Replica columns.

    Returns
    -------
    mean, sd, sem : tuple of numpy.ndarray
        ``sd`` and ``sem`` are zero when only one replica is present.
    """
    values = frame[list(columns)]
    mean = values.mean(axis=1).to_numpy(float)
    if len(columns) > 1:
        sd = values.std(axis=1, ddof=1).to_numpy(float)
        sem = sd / np.sqrt(len(columns))
    else:
        sd = np.zeros_like(mean)
        sem = np.zeros_like(mean)
    return mean, sd, sem


def equimolar_surfaces(z, rho, bulk_halfwidth):
    """Equimolar dividing surfaces for a film with negligible vapor density.

    This is not the half-density surface. The liquid density is the mean
    over the same 15 Å bulk window used everywhere else.

    Parameters
    ----------
    z : array_like
        Bin centers in angstroms.
    rho : array_like
        Mass density in g cm^-3. This should be the unsmoothed profile.
    bulk_halfwidth : float
        Half-width, about the density-weighted center, used to estimate
        the liquid density.

    Returns
    -------
    dict
        Center, liquid density, integrated density, thickness, and the
        lower and upper surfaces.
    """
    z = np.asarray(z, dtype=float)
    rho = np.asarray(rho, dtype=float)
    order = np.argsort(z)
    z = z[order]
    rho = rho[order]
    dz = float(np.median(np.diff(z)))
    integral = float(np.sum(rho) * dz)
    center = float(np.sum(z * rho) * dz / integral)
    bulk = (z >= center - bulk_halfwidth) & (z <= center + bulk_halfwidth)
    if not np.any(bulk):
        raise RuntimeError("No bins fall in the window used for the liquid density.")
    rho_liquid = float(np.mean(rho[bulk]))
    thickness = integral / rho_liquid
    return {
        "z_center": center,
        "rho_liquid": rho_liquid,
        "integral": integral,
        "thickness": thickness,
        "z_lower": center - 0.5 * thickness,
        "z_upper": center + 0.5 * thickness,
    }


def plot_profiles(frame, surfaces, destination, title):
    """Draw density and cosine on opposing horizontal axes.

    Parameters
    ----------
    frame : pandas.DataFrame
        Columns ``z``, ``density_smooth``, and ``cos_smooth``.
    surfaces : dict
        Equimolar and half-density surfaces.
    destination : pathlib.Path
        PNG output path.
    title : str
        Figure title.

    Returns
    -------
    None
        The figure is written to ``destination``.
    """
    figure, axis = plt.subplots(figsize=(10, 6))
    cosine_axis = axis.twiny()
    axis.plot(frame["density_smooth"], frame["z"], linewidth=2.5, color="blue", label="Density", zorder=3)
    cosine_axis.plot(
        frame["cos_smooth"],
        frame["z"],
        linewidth=2.5,
        color="red",
        label=r"$\langle \cos\theta \rangle$",
        zorder=4,
    )
    axis.axhline(0.0, color="black", linestyle="--", alpha=0.5, linewidth=1.20)
    if "half_lower" in surfaces:
        axis.axhline(surfaces["half_lower"], color="green", linestyle="-", linewidth=1.4, label="Half-density surface")
        axis.axhline(surfaces["half_upper"], color="green", linestyle="-", linewidth=1.4)
    axis.axhline(surfaces["z_lower"], color="black", linestyle="--", linewidth=1.8, label="Equimolar surface")
    axis.axhline(surfaces["z_upper"], color="black", linestyle="--", linewidth=1.8)
    axis.axhspan(surfaces["z_lower"], surfaces["z_upper"], alpha=0.08, color="gray", label="Equimolar slab")
    axis.axhline(surfaces["z_center"], color="black", linestyle=":", linewidth=1.20, label="Slab center")
    axis.set_xlabel(r"Density (g cm$^{-3}$)", color="blue")
    cosine_axis.set_xlabel(r"$\langle \cos\theta \rangle$", color="red")
    axis.set_ylabel(r"$z$ ($\mathrm{\AA}$)")
    axis.set_ylim(float(frame["z"].min()), float(frame["z"].max()))
    axis.yaxis.set_major_locator(MultipleLocator(20))
    axis.yaxis.set_minor_locator(MultipleLocator(10))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:+.0f}"))
    axis.set_xlim(0.0, max(1.2, float(frame["density_smooth"].max()) * 1.10))
    cosine_axis.set_xlim(-0.5, 0.5)
    axis.grid(axis="both", linestyle="--", alpha=0.35)
    axis.set_title(title)
    handles, labels = axis.get_legend_handles_labels()
    extra_handles, extra_labels = cosine_axis.get_legend_handles_labels()
    axis.legend(handles + extra_handles, labels + extra_labels, fontsize=10, loc="center left", bbox_to_anchor=(1.02, 0.5))
    summary = (
        rf"$\rho_l$: {surfaces['rho_liquid']:.3f} g/cm$^3$"
        "\n"
        rf"$z_c$: {surfaces['z_center']:.2f} $\mathrm{{\AA}}$"
        "\n"
        rf"$z_{{eq}}^-$: {surfaces['z_lower']:.2f} $\mathrm{{\AA}}$"
        "\n"
        rf"$z_{{eq}}^+$: {surfaces['z_upper']:.2f} $\mathrm{{\AA}}$"
        "\n"
        rf"$h_G$: {surfaces['thickness']:.2f} $\mathrm{{\AA}}$"
    )
    axis.text(
        1.02,
        0.23,
        summary,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "black", "alpha": 0.95},
    )
    figure.tight_layout()
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main(argv=None):
    """Average replica profiles and draw the equimolar surfaces.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments, excluding the program name.

    Returns
    -------
    int
        Process status code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--density", nargs="+", required=True, type=Path)
    parser.add_argument("--cosine", nargs="+", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("density_and_cos_theta_z_profile.png"))
    parser.add_argument("--lx", type=float, default=DEFAULT_LX_A, help="Unused. cpptraj number density is already per volume.")
    parser.add_argument("--ly", type=float, default=DEFAULT_LY_A, help="Unused. cpptraj number density is already per volume.")
    parser.add_argument("--temperature", type=float, default=None, help="If 300, the bulk density must lie within 0.10 g/cm^3 of 1.")
    parser.add_argument("--bulk-halfwidth", type=float, default=None)
    parser.add_argument("--title", default="OPC water density and dipole orientation")
    args = parser.parse_args(argv)
    densities = [load_density_file(path) for path in args.density]
    cosines = [load_cos_file(path) for path in args.cosine]
    density = densities[0].rename(columns={"density_g_cm3": "density_r1", "number_density_A3": "number_density_r1"})
    for index, frame in enumerate(densities[1:], start=2):
        density = density.merge(
            frame.rename(columns={"density_g_cm3": f"density_r{index}", "number_density_A3": f"number_density_r{index}"}),
            on="z",
            how="inner",
        )
    density_columns = [name for name in density.columns if name.startswith("density_r")]
    mean, _sd, _sem = _mean_sd(density, density_columns)
    density["density_mean"] = mean
    cosine = cosines[0].rename(columns={"mean_cos_theta": "cos_r1"})
    for index, frame in enumerate(cosines[1:], start=2):
        cosine = cosine.merge(frame.rename(columns={"mean_cos_theta": f"cos_r{index}"}), on="z", how="inner")
    cosine = cosine.replace([np.inf, -np.inf], np.nan).sort_values("z")
    cosine_columns = [name for name in cosine.columns if name.startswith("cos_r")]
    cosine_mean, _sd, _sem = _mean_sd(cosine, cosine_columns)
    cosine["cos_mean"] = pd.Series(cosine_mean, index=cosine.index).interpolate(limit_direction="both").fillna(0.0)
    density = density.sort_values("z")
    density["cos_mean"] = np.interp(density["z"], cosine["z"], cosine["cos_mean"], left=0.0, right=0.0)
    density["density_smooth"] = np.clip(
        smooth_profile(density["density_mean"].to_numpy(), density["z"].to_numpy()),
        0.0,
        None,
    )
    density["cos_smooth"] = smooth_profile(density["cos_mean"].to_numpy(), density["z"].to_numpy())
    halfwidth = BULK_HALFWIDTH_A if args.bulk_halfwidth is None else args.bulk_halfwidth
    surfaces = equimolar_surfaces(density["z"], density["density_mean"], halfwidth)
    centered = density["z"].to_numpy(float) - surfaces["z_center"]
    order = np.argsort(centered)
    half = half_density_surfaces(
        centered[order], density["density_smooth"].to_numpy(float)[order], halfwidth
    )
    surfaces["half_lower"] = half["z_lower"] + surfaces["z_center"]
    surfaces["half_upper"] = half["z_upper"] + surfaces["z_center"]
    rho_core = float(half["rho_bulk"])
    if args.temperature is not None and abs(args.temperature - 300.0) <= 1.0:
        if not 0.90 <= rho_core <= 1.10:
            raise RuntimeError(
                f"300 K bulk density is {rho_core:.3f} g/cm^3, not about 1. "
                "The second column must be the cpptraj number density in A^-3."
            )
    elif not 0.80 <= rho_core <= 1.20:
        raise RuntimeError(
            f"Bulk density is {rho_core:.3f} g/cm^3. "
            "A cpptraj number-density profile of this film is about 1 g/cm^3."
        )
    print(f"Density-weighted center: {surfaces['z_center']:.4f} A")
    print(f"Bulk density:            {rho_core:.6f} g/cm^3")
    print(f"Liquid density:          {surfaces['rho_liquid']:.6f} g/cm^3")
    print(f"Equimolar thickness:     {surfaces['thickness']:.4f} A")
    print(f"Lower surface:           {surfaces['z_lower']:.4f} A")
    print(f"Upper surface:           {surfaces['z_upper']:.4f} A")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot_profiles(density, surfaces, args.out, args.title)
    density.to_csv(args.out.with_suffix(".csv"), index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
