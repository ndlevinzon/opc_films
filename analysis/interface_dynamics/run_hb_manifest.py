#!/usr/bin/env python3
"""Run the hydrogen-bond analysis for one project-manifest row.

The output directory is ``results/hbond/<row_id>``. A hold or an endpoint is run
with ``--hold`` so the cooling-window lag cap is not applied.
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.manifest import load_manifest, result_directory, select_row, trajectory_list
from opcfilm.config import CONFIG


def main(argv=None):
    """Run ``hbond_v3.py`` on one manifest row.

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
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--row", type=int, required=True)
    parser.add_argument("--origins", type=int, default=16)
    parser.add_argument("--nref-region", type=int, default=CONFIG.dynamics.references_per_region)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--v2-root", default=None)
    args = parser.parse_args(argv)
    row = select_row(load_manifest(args.manifest), args.row)
    repo = Path(__file__).resolve().parents[2]
    out = result_directory(repo, "hbond", row)
    if (out / "hb_metadata.json").exists() and not args.overwrite:
        print("SKIP", out)
        return 0
    out.mkdir(parents=True, exist_ok=True)
    call = [
        sys.executable,
        str(Path(__file__).with_name("hbond_v3.py")),
        "--top", row["topology"],
        "--out", str(out),
        "--origins", str(args.origins),
        "--nref-region", str(args.nref_region),
    ]
    for path in trajectory_list(row):
        call.extend(["--traj", str(path)])
    if row["kind"] in ("hold", "endpoint"):
        call.append("--hold")
    events = result_directory(repo, "regional_dyn", row) / "event_samples.csv"
    if not events.exists() and args.v2_root:
        events = Path(args.v2_root) / row["replica"] / "ramp_nvt" / "event_samples.csv"
    if events.exists():
        call.extend(["--v2-events", str(events)])
    subprocess.run(call, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
