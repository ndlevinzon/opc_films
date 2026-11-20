#!/usr/bin/env python3
"""Run the free-OH analysis for one row of a trajectory manifest."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.constants import SEEDS
from common.manifest import load_manifest, select_row, trajectory_list
from common.trajectory import frames_skipped
from opcfilm.config import CONFIG


def output_directory(outroot, row):
    """Return the directory for one manifest row.

    Parameters
    ----------
    outroot : pathlib.Path
        Root passed as ``--outroot``. The Snakefile passes ``results/defects``.
    row : dict
        One manifest row. ``row_id`` distinguishes holds that share a replica.

    Returns
    -------
    pathlib.Path
        ``<outroot>/<row_id>``.
    """
    return Path(outroot) / str(row["row_id"])


def main(argv=None):
    """Launch ``03_analyze_free_oh.py`` on the requested manifest row.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments, excluding the program name.

    Returns
    -------
    int
        Process status code. Zero if the row is already complete and
        ``--overwrite`` was not set.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--row", type=int, required=True, help="1-based manifest row.")
    parser.add_argument("--outroot", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--nref", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=SEEDS["hydrogen_bond_free_oh"], help="Base seed. The row index is added before the analysis.")
    parser.add_argument("--outward-cos", type=float, default=0.5)
    parser.add_argument("--core-halfwidth", type=float, default=CONFIG.regions.bulk_halfwidth_A)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    row = select_row(load_manifest(args.manifest), args.row)
    paths = trajectory_list(row)
    topology = Path(str(row["topology"]))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing or not topology.is_file():
        raise SystemExit(f"Missing input: {missing or topology}")

    destination = output_directory(args.outroot, row)
    finished = [
        destination / "awi_0-6" / "free_oh_metadata.json",
        destination / "awi_3-12" / "free_oh_metadata.json",
    ]
    if all(path.exists() for path in finished) and not args.overwrite:
        print(f"Skipping complete output {destination}")
        return 0

    analyzer = Path(__file__).with_name("03_analyze_free_oh.py")
    command = [
        sys.executable,
        str(analyzer),
        "--top",
        str(topology),
        "--out",
        str(destination),
        "--frames",
        str(args.frames),
        "--nref",
        str(args.nref),
        "--seed",
        str(args.seed + args.row),
        "--outward-cos",
        str(args.outward_cos),
        "--core-halfwidth",
        str(args.core_halfwidth),
        "--start-frame",
        str(frames_skipped(row.get("discard_ps", ""), row.get("frame_ps", ""))),
    ]
    for path in paths:
        command.extend(["--traj", str(path)])
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
