#!/usr/bin/env python3
"""Overlay in-plane structure factors and write their mean and standard deviation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from opcfilm.plotting.conventions import install_figure_conventions
install_figure_conventions()  # temperature axes hot to cold; legends outside panels
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.uncertainty import curve_band, curve_caption


def load_curve(path):
    """Read one ``q`` and ``S(q)`` curve.

    Parameters
    ----------
    path : path-like
        CSV with columns ``q_Ainv`` and ``Sq2D``.

    Returns
    -------
    pandas.DataFrame
        The two columns, with ``Sq2D`` renamed to ``path`` so several
        curves can be merged.
    """
    frame = pd.read_csv(path)
    return frame[["q_Ainv", "Sq2D"]].rename(columns={"Sq2D": str(path)})


def average_curves(paths):
    """Average structure factors on their common wavevector grid.

    Parameters
    ----------
    paths : sequence of str
        CSV paths produced by ``02_sq2_fft.py``.

    Returns
    -------
    q, mean, sd, merged : tuple
        Wavevector, mean intensity, sample standard deviation, and the
        merged table.
    """
    curves = [load_curve(path) for path in paths]
    merged = curves[0]
    for curve in curves[1:]:
        merged = merged.merge(curve, on="q_Ainv", how="inner")
    q = merged["q_Ainv"].to_numpy()
    values = merged.drop(columns="q_Ainv").to_numpy()
    mean = np.nanmean(values, axis=1)
    sd = np.nanstd(values, axis=1, ddof=1) if values.shape[1] > 1 else np.zeros_like(mean)
    return q, mean, sd, merged


def main(argv=None):
    """Plot the supplied curves, their mean, and a two-standard-deviation band.

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
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("-o", "--out", default="Sq2D_compare.png")
    parser.add_argument("--band-sd", type=float, default=1.0, help="Replica-band width in standard deviations.")
    parser.add_argument("--qstar-json", default=None,
                        help="outputs/qstar.json. Marks the adopted q* and the measured mean peak.")
    args = parser.parse_args(argv)
    plt.figure(figsize=(8, 6))
    for path in args.inputs:
        frame = pd.read_csv(path)
        plt.plot(frame["q_Ainv"], frame["Sq2D"], color="0.6", alpha=0.45, linewidth=0.9)
    q, mean, sd, _merged = average_curves(args.inputs)
    plt.plot(q, mean, color="black", linewidth=2.2, label="Mean of the rows")
    low, high = curve_band(mean, sd, args.band_sd)
    plt.fill_between(q, low, high, color="0.5", alpha=0.22, lw=0, label=curve_caption(args.band_sd))
    marks = {}
    if args.qstar_json and Path(args.qstar_json).is_file():
        payload = json.loads(Path(args.qstar_json).read_text())
        adopted = float(payload["qstar_adopted_Ainv"])
        measured = float(payload["qstar_measured_mean_Ainv"])
        marks = {"qstar_adopted_Ainv": adopted, "qstar_measured_mean_Ainv": measured,
                 "qstar_measured_sd_Ainv": payload.get("qstar_measured_sd_Ainv")}
        plt.axvline(adopted, color="black", ls=":", lw=1.1, label=rf"Adopted $q^*$ = {adopted:.3f} Å$^{{-1}}$")
        if abs(measured - adopted) > 1e-6:
            plt.axvline(measured, color="0.45", ls="-.", lw=1.0,
                        label=rf"Measured peak = {measured:.3f} Å$^{{-1}}$")
    plt.xlim(0.5, 4.0)
    plt.xlabel(r"$q$ ($\AA^{-1}$)")
    plt.ylabel(r"$S^{\parallel}(q)$")
    plt.title("In-plane oxygen structure factor")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    plt.savefig(str(Path(args.out).with_suffix(".pdf")))
    plt.close()
    table = pd.DataFrame({"q_Ainv": q, "Sq_mean": mean, "Sq_sd": sd})
    table_path = args.out.replace(".png", "_mean.csv")
    table.to_csv(table_path, index=False)
    if marks:
        Path(args.out.replace(".png", "_qstar.json")).write_text(json.dumps(marks, indent=2))
    print(f"Wrote {args.out} and {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
