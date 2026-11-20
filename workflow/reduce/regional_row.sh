#!/bin/bash
# One regional-dynamics row. Holds and endpoints pass --hold.
set -euo pipefail
cd "$(dirname "$0")/../.."
python - "$1" <<'PY'
import subprocess
import sys

import pandas as pd

index = int(sys.argv[1])
frame = pd.read_csv("config/manifest.tsv", sep="\t", dtype=str, keep_default_na=False)
record = frame.iloc[index - 1]
command = [
    sys.executable,
    "analysis/interface_dynamics/analyze_v2.py",
    "--manifest", "config/manifest.tsv",
    "--row", str(index),
]
if record["kind"] != "ramp":
    command.append("--hold")
subprocess.check_call(command)
PY
