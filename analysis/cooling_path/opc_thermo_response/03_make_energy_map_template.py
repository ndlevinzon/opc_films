
#!/usr/bin/env python3
"""Match OPC cooling trajectories to AMBER MDOUT files."""

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from common.manifest import load_manifest, trajectory_list

ap = argparse.ArgumentParser()
ap.add_argument("--density-manifest", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

pattern = re.compile(
    r"chunk_(\d+)_(\d+)_to_(\d+)K",
    re.IGNORECASE,
)

def chunk_key(path):
    match = pattern.search(Path(path).name)
    if match is None:
        return None
    return tuple(int(v) for v in match.groups())

rows = load_manifest(args.density_manifest)

mapping = []
review = []

for row in rows:
    paths = trajectory_list(row)
    traj = paths[0] if paths else Path()
    status = ""
    mdout = ""

    if row["kind"] != "ramp":
        status = "SKIPPED_HOLD"
    else:
        outdir = traj.parent.parent / "OUT"

        # Search actual AMBER MDOUT extensions.
        candidates = sorted(
            list(outdir.glob("*.mdout")) +
            list(outdir.glob("*.out"))
        )

        # Prefer identical filename stems.
        exact = [p for p in candidates if p.stem == traj.stem]

        if len(exact) == 1:
            mdout = str(exact[0])
            status = "EXACT_STEM"
        else:
            key = chunk_key(traj)
            matches = [
                p for p in candidates
                if key is not None and chunk_key(p) == key
            ]

            if len(matches) == 1:
                mdout = str(matches[0])
                status = "TEMPERATURE_AND_CHUNK"
            elif len(matches) > 1:
                status = "AMBIGUOUS"
            else:
                status = "NOT_FOUND"

    mapping.append({
        "trajectory": str(traj),
        "mdout": mdout,
    })

    review.append({
        "trajectory": str(traj),
        "mdout": mdout,
        "status": status,
    })

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)

with out.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["trajectory", "mdout"],
        delimiter="\t",
    )
    writer.writeheader()
    writer.writerows(mapping)

review_path = out.with_name("energy_map_review.tsv")

with review_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["trajectory", "mdout", "status"],
        delimiter="\t",
    )
    writer.writeheader()
    writer.writerows(review)

from collections import Counter
print("Mapping complete:", out)
print("Results:", dict(Counter(r["status"] for r in review)))
