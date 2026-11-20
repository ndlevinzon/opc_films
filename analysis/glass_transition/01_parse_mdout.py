#!/usr/bin/env python3
"""Write Amber mdout thermodynamic records to a CSV file."""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.mdout import parse_file


def expand_inputs(patterns):
    """Expand file names and glob patterns to existing files.

    Parameters
    ----------
    patterns : sequence of str
        Literal paths or glob patterns.

    Returns
    -------
    list of str
        Sorted unique paths that exist.
    """
    files = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        files.extend(matches if matches else [pattern])
    return sorted({path for path in files if os.path.isfile(path)})


def main(argv=None):
    """Parse one or more mdout files and write a combined CSV.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments, excluding the program name.

    Returns
    -------
    int
        Process status code.
    """
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patterns", nargs="+", help="mdout file(s) or glob pattern(s).")
    parser.add_argument("-o", "--out", required=True)
    parser.add_argument("--rate", type=float, default=None, help="Cooling rate in K/ns.")
    parser.add_argument("--temperature", type=float, default=None, help="Isothermal target in K.")
    parser.add_argument("--replica", default=None)
    args = parser.parse_args(argv)
    files = expand_inputs(args.patterns)
    if not files:
        raise SystemExit("No input files found.")
    rows = []
    for path in files:
        rows.extend(parse_file(path))
    frame = pd.DataFrame(rows)
    if args.rate is not None:
        frame["rate_K_per_ns"] = args.rate
    if args.temperature is not None:
        frame["target_T_K"] = args.temperature
    if args.replica is not None:
        frame["replica"] = args.replica
    preferred = [
        "source",
        "nstep",
        "time_ps",
        "temp_K",
        "press_bar",
        "Etot",
        "EKtot",
        "EPtot",
        "volume_A3",
        "density_g_cm3",
        "rate_K_per_ns",
        "target_T_K",
        "replica",
    ]
    columns = [name for name in preferred if name in frame.columns]
    columns += [name for name in frame.columns if name not in columns]
    frame = frame[columns]
    frame.to_csv(args.out, index=False)
    print(f"Wrote {len(frame)} records to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
