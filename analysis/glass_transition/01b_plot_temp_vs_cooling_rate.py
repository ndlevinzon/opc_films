#!/usr/bin/env python3
"""Plot the thermostat temperature of each cooling protocol.

Each cooling rate is stored as three replica CSVs produced by
``01_parse_mdout.py``. The file names follow
``ramp_<tag>_r<replica>.csv``, where the tag uses ``p`` for the decimal
point (``4p44``, ``11p1``, ``22p2``, ``44p4``, ``111``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from opcfilm.plotting.conventions import install_figure_conventions
install_figure_conventions()  # temperature axes hot to cold; legends outside panels
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.rates import PHYSICAL_RATES, file_tag

RATES = [(rate, file_tag(rate)) for rate in reversed(PHYSICAL_RATES)]


def replica_table(path, replica):
    """Read time and temperature from one mdout CSV.

    Parameters
    ----------
    path : pathlib.Path
        CSV with ``time_ps`` and ``temp_K``.
    replica : int
        Replica index recorded in the returned table.

    Returns
    -------
    pandas.DataFrame
        Columns ``time_ns``, ``temp_K``, and ``replica``.
    """
    frame = pd.read_csv(path)
    for column in ("time_ps", "temp_K"):
        if column not in frame.columns:
            raise RuntimeError(f"{path} does not contain {column}")
    table = frame[["time_ps", "temp_K"]].dropna().copy()
    table["time_ns"] = table["time_ps"] / 1000.0
    table["replica"] = replica
    return table


def plot_protocols(outdir, replicas):
    """Plot faint replica traces and the mean temperature of each rate.

    Parameters
    ----------
    outdir : pathlib.Path
        Directory that contains the replica CSVs and receives the figure.
    replicas : sequence of int
        Replica indices, usually ``1, 2, 3``.

    Returns
    -------
    pathlib.Path
        Written PNG path.
    """
    figure, axis = plt.subplots(figsize=(8.0, 5.5))
    for rate, tag in RATES:
        tables = []
        for replica in replicas:
            path = outdir / f"ramp_{tag}_r{replica}.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            table = replica_table(path, replica)
            tables.append(table)
            axis.plot(table["time_ns"], table["temp_K"], linewidth=0.7, alpha=0.20)
        combined = pd.concat(tables, ignore_index=True)
        mean = combined.groupby("time_ns", as_index=False).agg(
            temp_mean=("temp_K", "mean"),
            temp_sd=("temp_K", "std"),
        )
        axis.plot(mean["time_ns"], mean["temp_mean"], linewidth=2.0, label=f"{rate:g} K/ns")
    axis.set_xlabel("Time (ns)")
    axis.set_ylabel("Temperature (K)")
    axis.set_title("OPC water cooling protocols")
    axis.legend(title="Cooling rate", frameon=False)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    destination = outdir / "temperature_vs_time.png"
    figure.savefig(destination, dpi=300)
    plt.close(figure)
    return destination


def main(argv=None):
    """Plot every cooling protocol found in ``--outdir``.

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
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    parser.add_argument("--replicas", nargs="+", type=int, default=[1, 2, 3])
    args = parser.parse_args(argv)
    destination = plot_protocols(args.outdir, args.replicas)
    print(f"Wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
