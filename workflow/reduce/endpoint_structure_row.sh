#!/bin/bash
# Structure of one endpoint (300 K or 78 K): local order and S(q) from
# local_order.py, density and dipole profiles from dipole_profile.py.
# The manifest discard is skipped. Frames are thinned because neighbouring
# frames of an endpoint are correlated.
set -euo pipefail
cd "$(dirname "$0")/../.."
python - "$1" <<'PY'
import subprocess
import sys
from pathlib import Path

from opcfilm.catalog import load_manifest, result_directory, select_row, trajectory_list

index = sys.argv[1]
row = select_row(load_manifest("config/manifest.tsv"), int(index))
if row["kind"] != "endpoint":
    raise SystemExit(f"Row {index} is a {row['kind']}, not an endpoint.")
destination = result_directory(Path("."), "endpoint_structure", row).resolve()
trajectories = [str(path) for path in trajectory_list(row)]
discard = row.get("discard_ps", "") or "0"
subprocess.check_call([
    sys.executable, "analysis/glass_structure/local_order.py",
    "--topology", row["topology"],
    "--trajectory", *trajectories,
    "--discard-ps", discard,
    "--stride", "250",
    "--out", str(destination / "local_order"),
])
subprocess.check_call([
    sys.executable, "analysis/glass_structure/dipole_profile.py",
    "--topology", row["topology"],
    "--trajectory", *trajectories,
    "--discard-ps", discard,
    "--stride", "50",
    "--dz", "1.0",
    "--out", str(destination / "dipole"),
])
PY
