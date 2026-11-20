#!/usr/bin/env python3
"""Write the one project manifest for ramps, holds, and endpoints.

Ramps are read from ``<root>/opc/<replica>/ramp_nvt/<rate>K_ns/TRAJ/``.
Holds and endpoints are read from ``opc/<replica>/hold/`` and
``opc/<replica>/endpoint/`` when those directories exist, and from
``--holds-tsv`` and ``--endpoints-tsv``. The topology defaults to
``<root>/build/opc_slab.prmtop``. Output is ``outputs/manifest.tsv``.
``outputs/density_manifest.tsv`` is the same table.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.manifest import (
    cutoff_note,
    discard_fraction_for,
    discover_rows,
    output_directory,
    read_cut,
    write_manifest,
)
from common.mdout import time_span_ps
from common.rates import PHYSICAL_RATES, REPLICAS, directory_code


def listed_rows(table, workflow, topology, default_kind):
    """Append trajectories named in a TSV that are not in the data tree.

    Parameters
    ----------
    table : pathlib.Path
        Tab-separated file with columns ``label``, ``replica``,
        ``trajectory``, and ``temperature_K``. Optional columns are
        ``kind``, ``rate_K_ns``, ``source_restart``, ``frame_ps``, and
        ``mdout``.
    workflow : pathlib.Path
        Directory under which ``outputs/`` is created.
    topology : pathlib.Path
        Amber topology recorded when the row does not name one.
    default_kind : str
        ``hold`` or ``endpoint``, used when the table has no ``kind`` column.

    Returns
    -------
    list of dict
        Manifest rows. Output directories encode ``kind``.
    """
    rows = []
    with table.open(newline="") as handle:
        for item in csv.DictReader(handle, delimiter="\t"):
            trajectory = Path(item["trajectory"])
            replica = item.get("replica", "r1")
            label = item.get("label", trajectory.stem)
            temperature = item["temperature_K"]
            kind = item.get("kind") or default_kind
            rate = item.get("rate_K_ns", "")
            mdout = item.get("mdout", "")
            cut = ""
            duration = item.get("duration_ps", "")
            if mdout:
                cut = f"{read_cut(mdout):g}"
                span = time_span_ps(mdout)
                if span is not None:
                    duration = f"{span:g}"
            rows.append({
                "kind": kind,
                "replica": replica,
                "rate_code": item.get("rate_code", ""),
                "rate_K_ns": rate,
                "temperature_start_K": temperature,
                "temperature_end_K": temperature,
                "temperature_K": temperature,
                "trajectory": str(trajectory.resolve()),
                "frame_ps": item.get("frame_ps", ""),
                "source_restart": item.get("source_restart", ""),
                "topology": str(Path(item.get("topology") or topology).resolve()),
                "cut_A": cut,
                "mdout": mdout,
                "duration_ps": duration,
                "discard_fraction": f"{discard_fraction_for(kind):g}",
                "outdir": str(output_directory(
                    workflow / "outputs", kind, replica, rate or "unset", temperature, label
                )),
                "label": label,
            })
    return rows


def main(argv=None):
    """Write ``config/manifest.tsv`` by calling ``opcfilm catalog build``.

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
    parser.add_argument("--root", type=Path, default=os.environ.get("OPC_DATA_ROOT"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "config" / "manifest.tsv",
    )
    args = parser.parse_args(argv)
    if args.root is None:
        parser.error("Pass --root or set OPC_DATA_ROOT.")
    import subprocess
    return subprocess.call([
        sys.executable, "-m", "opcfilm.cli", "catalog", "build",
        "--root", str(args.root), "--out", str(args.out), "--allow-dirty",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
