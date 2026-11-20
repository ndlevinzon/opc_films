#!/bin/bash
# Energy per ramp window and the arrest fit. The comparison with T_g^dyn is compare_arrest.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p results/energy results/cooling_summary
python - <<'PY'
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

frame = pd.read_csv("config/manifest.tsv", sep="\t", dtype=str, keep_default_na=False)
ramps = frame[frame.kind == "ramp"]
if ramps.empty:
    raise SystemExit("The manifest has no ramp rows.")
groups = defaultdict(list)
for row in ramps.to_dict("records"):
    groups[(row["replica"], row["rate_K_per_ns"])].extend(
        part for part in str(row.get("mdout", "")).split(";") if part
    )
for (replica, rate), paths in sorted(groups.items()):
    if not paths:
        raise SystemExit(f"{replica} at {rate} K/ns has no mdout.")
    destination = Path(f"results/energy/ramp_{rate}_{replica}.csv")
    subprocess.check_call([
        sys.executable, "analysis/glass_transition/01_parse_mdout.py",
        "-o", str(destination), "--rate", rate, "--replica", replica, *paths,
    ])
topology = ramps.iloc[0]["topology"]
pairs = []
for (replica, rate), paths in sorted(groups.items()):
    pairs.append(f"{rate}=results/energy/ramp_{rate}_{replica}.csv")
subprocess.check_call([
    sys.executable, "analysis/glass_transition/06_plot_ramp_energy.py",
    "--topology", topology, "--out-prefix", "results/energy/ramp", *pairs,
])
subprocess.check_call([
    sys.executable, "analysis/glass_transition/06b_fit_energy_tg.py",
    "-i", "results/energy/ramp_replica_binned.csv",
    "--topology", topology, "--out-prefix", "results/energy/arrest",
])
subprocess.check_call([
    sys.executable,
    "analysis/cooling_path/opc_thermo_response/03_make_energy_map_template.py",
    "--density-manifest", "config/manifest.tsv",
    "--out", "results/energy/energy_map.tsv",
])
PY
python analysis/cooling_path/04_summarize.py --manifest config/manifest.tsv --out results/cooling_summary
