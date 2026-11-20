#!/usr/bin/env python3
"""Build a hydrogen-bond manifest from the project manifest.

Pass ``--project-manifest`` to copy ``outputs/manifest.tsv``. Without it,
the same discovery is used: ramps under ``ramp_nvt``, holds under
``opc/<replica>/hold/``, and endpoints under ``opc/<replica>/endpoint/``.
Each row keeps ``kind``. ``temp_mid_K`` is the ramp midpoint or the hold
setpoint. This archive does not contain those hold or endpoint files.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.manifest import COLUMNS, discover_rows as project_rows, load_manifest
from common.rates import PHYSICAL_RATES, REPLICAS, directory_code

_COLUMNS = ["temp_mid_K", *COLUMNS]


def discover_rows(root, workflow, topology, replicas, rate_codes):
    """List ramp, hold, and endpoint trajectories.

    Parameters
    ----------
    root : pathlib.Path
        Directory that contains ``opc/``.
    workflow : pathlib.Path
        Directory under which output paths are created.
    topology : pathlib.Path
        Amber topology shared by every row.
    replicas : sequence of str
        Replica directory names, for example ``r1``.
    rate_codes : sequence of int
        Rounded cooling-rate directory codes.

    Returns
    -------
    list of dict
        Project-manifest rows with ``temp_mid_K`` copied from
        ``temperature_K``.
    """
    rows = project_rows(root, workflow, topology, replicas, rate_codes)
    for row in rows:
        row["temp_mid_K"] = row["temperature_K"]
    return rows


def write_manifest(rows, destination):
    """Write manifest rows as a tab-separated table.

    Parameters
    ----------
    rows : sequence of dict
        Rows from :func:`discover_rows`.
    destination : pathlib.Path
        Output path. Parent directories are created.

    Returns
    -------
    None
        The table is written to ``destination``.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=_COLUMNS, delimiter="\t", extrasaction="ignore", restval=""
        )
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    """Parse command-line arguments and write the manifest.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments, excluding the program name. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process status code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=os.environ.get("OPC_DATA_ROOT"),
        help="Directory containing opc/ and build/. Defaults to OPC_DATA_ROOT.",
    )
    parser.add_argument(
        "--topology",
        type=Path,
        default=None,
        help="Amber prmtop. Defaults to <root>/build/opc_slab.prmtop.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "config" / "manifest.tsv",
        help="Output TSV path. The project manifest is config/manifest.tsv.",
    )
    parser.add_argument("--project-manifest", type=Path, default=None)
    parser.add_argument("--workflow", type=Path, default=Path.cwd())
    parser.add_argument("--replicas", nargs="+", default=list(REPLICAS))
    parser.add_argument(
        "--rates",
        nargs="+",
        type=float,
        default=list(PHYSICAL_RATES),
        help="Physical cooling rates in K/ns.",
    )
    args = parser.parse_args(argv)
    if args.root is None and args.project_manifest is None:
        parser.error("Pass --root or set OPC_DATA_ROOT.")
    if args.root is None:
        print(f"The project manifest is {args.project_manifest}.")
        return 0
    import subprocess
    return subprocess.call([
        sys.executable, "-m", "opcfilm.cli", "catalog", "build",
        "--root", str(args.root), "--out", str(args.out), "--allow-dirty",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
