#!/bin/bash
# One S(q) row. The argument is the 1-based manifest line.
# Every segment is passed, and 02_sq2_fft.py drops the hold equilibration.
set -euo pipefail
cd "$(dirname "$0")/../.."
python - "$1" <<'PY'
import subprocess
import sys

from opcfilm.catalog import load_manifest, select_row, trajectory_list

index = sys.argv[1]
row = select_row(load_manifest("config/manifest.tsv"), int(index))
destination = f"results/qstar_row/{index}"
command = [
    sys.executable,
    "analysis/glass_transition/02_sq2_fft.py",
    "-p", row["topology"],
    "-o", f"{destination}/sq.csv",
    "--qstar-json", f"{destination}/qstar_row.json",
]
for path in trajectory_list(row):
    command.extend(["-x", str(path)])
subprocess.check_call(command)
PY
