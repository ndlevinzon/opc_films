#!/bin/bash
# One F_s row. The argument is the 1-based manifest line.
set -euo pipefail
cd "$(dirname "$0")/../.."
python - "$1" <<'PY'
import subprocess
import sys
from pathlib import Path

from opcfilm.catalog import load_manifest, result_directory, select_row

index = sys.argv[1]
row = select_row(load_manifest("config/manifest.tsv"), int(index))
destination = result_directory(Path("."), "fs_tau", row)
destination.mkdir(parents=True, exist_ok=True)
subprocess.check_call([
    sys.executable,
    "analysis/glass_transition/03_fs_alpha.py",
    "--manifest", "config/manifest.tsv",
    "--row", index,
    "--curve-out", str(destination / "fs_curve.csv"),
    "--summary-out", str(destination / "tau_summary.csv"),
])
PY
