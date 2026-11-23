#!/usr/bin/env python3
"""Gather every table the SI quotes into one folder, by SI table label.

Run from the repository root after the figures rule. For each SI table the
script copies the pipeline outputs it is filled from into
``results/si_tables/<label>/`` and writes the reductions that no other script
makes:

* ``tab_si_inventory/inventory.csv``           rows, segments, durations per kind and temperature;
* ``tab_si_qstar/qstar_rows.csv``              the per-row peak JSON files as one table;
* ``tab_si_chill/band_order_summary_by_replica.csv`` and ``..._summary.csv``
  CHILL+ fractions and mean q, Sk, LSI per band at 300 and 78 K;
* ``tab_si_bragg/bragg_*_by_replica.csv``      Bragg screening of the endpoints;
* ``tab_si_software/software_versions.csv``    package versions and the
  cpptraj and pmemd version lines found in logs and mdout headers.

``results/si_tables/INDEX.md`` lists, for every table, the files that were
found and the ones that were not. Zip ``results/si_tables`` to send the
numbers on.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

W = "results/figures/work"
# label -> glob patterns relative to the repository root
SOURCES = {
    "tab:si_inventory": ["config/manifest.tsv"],
    "tab:si_hold_drift": ["results/cooling_summary/hold_drift_diagnostics.csv",
                          "results/cooling_summary/missing_outputs.txt"],
    "tab:si_capillary": ["results/capillary_summary/capillary_by_temperature.csv",
                         "results/capillary_summary/capillary_by_replica.csv",
                         "results/capillary_summary/capillary_table_rows.tex"],
    "tab:si_rate_summary": ["results/cooling_summary/density_minimum_summary.csv",
                            "results/cooling_summary/density_minimum_by_replica.csv",
                            "results/cooling_summary/rate_temperature_mean_sd.csv",
                            "results/thermo/thermo_response_replica_summary.csv",
                            "results/energy/ramp_lowT_fast_vs_slow.csv",
                            "results/energy/ramp_energetic_arrest_summary.csv"],
    "tab:si_chi": ["results/cooling_summary/rate_response_intervals_summary.csv",
                   "results/cooling_summary/rate_response_intervals_by_replica.csv"],
    "tab:si_temperature_resolved": ["results/cooling_summary/rate_temperature_mean_sd.csv",
                                    f"{W}/energetic_stabilization/fig2d_mean_sd_per_water.csv",
                                    "results/energy/ramp_mean_sd_per_water.csv",
                                    "results/thermo/thermo_response_replica_summary.csv"],
    "tab:si_derivative_window": [f"{W}/panels/SI_derivative_window.csv",
                                 "results/thermo/thermo_derivatives_by_replica.csv",
                                 "results/thermo/thermo_derivatives_window5_by_replica.csv"],
    "tab:si_qstar": ["outputs/qstar.json", "results/figures/si/SI_Sq2D_compare_mean.csv",
                     "results/figures/si/SI_Sq2D_compare_qstar.json"],
    "tab:si_relaxation": ["results/tg/tau_all.csv", "results/tg/tau_all_whole.csv",
                          "results/tg/4.44_temperature_summary.csv",
                          "results/figures/si/SI_tau_stationarity.csv",
                          "results/figures/si/SI_fs_curves_all_replica_interpolated_summary.csv"],
    "tab:si_arrhenius": ["results/tg/4.44_fit_summary.csv",
                         "results/energy/compare_dynamic_bootstrap_summary.csv"],
    "tab:si_arrhenius_window": ["results/tg/4.44_arrhenius_window_sensitivity.csv",
                                "results/tg/4.44_tau230_half_frame.csv"],
    "tab:si_tg_rates": ["results/tg/*_fit_summary.csv", "results/tg/tg_dyn_vs_rate.csv",
                        "results/energy/ramp_energetic_arrest_summary.csv",
                        "results/energy/ramp_energetic_arrest_by_replica.csv",
                        "results/energy/arrest_summary.csv", "results/energy/arrest_by_replica.csv",
                        "results/energy/compare_comparison.csv"],
    "tab:si_tg_compare": ["results/energy/compare_rate_dependence_fits.csv",
                          "results/energy/compare_dynamic_bootstrap_summary.csv",
                          "results/energy/compare_comparison.csv",
                          "results/energy/ramp_lowT_fast_vs_slow.csv",
                          "results/energy/arrest_window_sensitivity.csv"],
    "tab:si_regional_tau": [f"{W}/regional_dynamics/tau_censoring_corrected.csv",
                            f"{W}/regional_dynamics/tau_summary_corrected.csv",
                            f"{W}/regional_dynamics/tau_summary.csv",
                            f"{W}/regional_dynamics/coverage.csv",
                            f"{W}/regional_dynamics/stationarity_screen.csv"],
    "tab:si_regional_contrast": [f"{W}/regional_contrast/Fig1C_region_contrast_summary.csv",
                                 f"{W}/regional_dynamics/decoupling_summary.csv",
                                 f"{W}/regional_dynamics/decoupling_paired_resolved.csv",
                                 f"{W}/regional_summary/V4_DTR_paired_delta_summary.csv",
                                 f"{W}/hb_onset/RELAX_region_contrast_summary.csv"],
    "tab:si_defects_regions": [f"{W}/defects/between_replica_summary.csv", f"{W}/defects/replica_means.csv"],
    "tab:si_defect_shares": [f"{W}/defects/paired_class_decomposition_bulk_minus_AWI.csv"],
    "tab:si_hb_onset": [f"{W}/hb_onset/HB_onset_temperature_lag_table.csv",
                        f"{W}/hb_onset/TABLE_HB_onset_compact.csv",
                        f"{W}/hb_onset/HB_onset_paired_replicas.csv",
                        f"{W}/hbond/hb_intermittent_replica.csv"],
    "tab:si_chill": [],
    "tab:si_bragg": [],
    "tab:si_loro": ["results/classifier/leave_one_replica_out.tsv", "results/classifier/classifier_model.json"],
    "tab:si_ftilde_half": ["results/classifier/half_point_summary.csv",
                           "results/classifier/half_point_by_replica.csv",
                           "results/classifier/hold_region_replica_means.csv",
                           "results/classifier/region_summary.tsv"],
    "tab:si_software": [],
}


def folder(out, label):
    return out / label.replace(":", "_")


def copy_sources(root, out):
    found, missing = {}, {}
    for label, patterns in SOURCES.items():
        destination = folder(out, label)
        destination.mkdir(parents=True, exist_ok=True)
        found[label], missing[label] = [], []
        for pattern in patterns:
            matches = sorted(root.glob(pattern))
            if not matches:
                missing[label].append(pattern)
            for path in matches:
                name = path.name
                if (destination / name).exists():
                    name = f"{path.parent.name}__{name}"
                shutil.copy2(path, destination / name)
                found[label].append(str(path.relative_to(root)))
    return found, missing


def manifest(root):
    return pd.read_csv(root / "config" / "manifest.tsv", sep="\t", dtype=str, keep_default_na=False)


def inventory(root, out):
    m = manifest(root)
    for column in ("temperature_K", "duration_ps", "discard_ps", "frame_ps", "cut_A"):
        m[column] = pd.to_numeric(m[column], errors="coerce")
    m["segments"] = m["files"].str.split(";").str.len()
    m["analyzed_ps"] = m["duration_ps"] - m["discard_ps"].fillna(0)
    rows = []
    for (kind, temperature), part in m.groupby(["kind", "temperature_K"]):
        rows.append({"kind": kind, "temperature_K": temperature, "rows": len(part),
                     "segments_min": part.segments.min(), "segments_max": part.segments.max(),
                     "duration_ns_min": part.duration_ps.min() / 1e3, "duration_ns_max": part.duration_ps.max() / 1e3,
                     "discard_ns_min": part.discard_ps.min() / 1e3, "discard_ns_max": part.discard_ps.max() / 1e3,
                     "analyzed_ns_min": part.analyzed_ps.min() / 1e3, "analyzed_ns_max": part.analyzed_ps.max() / 1e3,
                     "frame_ps": ";".join(sorted({f"{v:g}" for v in part.frame_ps.dropna()})),
                     "cut_A": ";".join(sorted({f"{v:g}" for v in part.cut_A.dropna()})),
                     "rates_K_per_ns": ";".join(sorted((set(part.rate_K_per_ns) | set(part.get("source_rate_K_per_ns", []))) - {""})),
                     "source_restart": ";".join(sorted(set(part.source_restart)))[:200]})
    pd.DataFrame(rows).to_csv(folder(out, "tab:si_inventory") / "inventory.csv", index=False)


def qstar_rows(root, out):
    m = manifest(root)
    rows = []
    for path in sorted(root.glob("results/qstar_row/*/qstar_row.json")):
        payload = json.loads(path.read_text())
        line = int(path.parent.name)
        record = {"manifest_line": line, "row_id": m.iloc[line - 1]["row_id"] if line <= len(m) else ""}
        record.update({k: v for k, v in payload.items() if not isinstance(v, (list, dict))})
        rows.append(record)
    if rows:
        pd.DataFrame(rows).to_csv(folder(out, "tab:si_qstar") / "qstar_rows.csv", index=False)
    return bool(rows)


def endpoint_tables(root, out):
    """CHILL+, band order, and Bragg tables of the 300 and 78 K endpoints."""
    m = manifest(root)
    ends = m[m.kind == "endpoint"]
    band, bragg_peaks, bragg_angle = [], [], []
    for _, row in ends.iterrows():
        directory = root / "results" / "endpoint_structure" / row["row_id"] / "local_order"
        tags = {"row_id": row["row_id"], "replica": row["replica"], "temperature_K": float(row["temperature_K"])}
        for name, bucket in (("band_order_summary.csv", band), ("water_bragg_peak_diagnostics.csv", bragg_peaks),
                             ("water_bragg_angular_anisotropy.csv", bragg_angle)):
            path = directory / name
            if path.is_file() and path.stat().st_size > 0:
                try:
                    bucket.append(pd.read_csv(path).assign(**tags))
                except pd.errors.EmptyDataError:
                    pass
    found = {}
    if band:
        table = pd.concat(band, ignore_index=True)
        table.to_csv(folder(out, "tab:si_chill") / "band_order_summary_by_replica.csv", index=False)
        values = [c for c in table.columns if c.startswith(("chill_fraction_", "mean_"))]
        summary = table.groupby(["temperature_K", "region"])[values].agg(["mean", "std"])
        summary.columns = [f"{a}_{b if b == 'mean' else 'sd'}" for a, b in summary.columns]
        summary.reset_index().to_csv(folder(out, "tab:si_chill") / "band_order_summary.csv", index=False)
        found["band"] = True
    for name, frames in (("bragg_peak_diagnostics_by_replica.csv", bragg_peaks),
                         ("bragg_angular_anisotropy_by_replica.csv", bragg_angle)):
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(folder(out, "tab:si_bragg") / name, index=False)
            found[name] = True
    return found


VERSION = re.compile(r"(cpptraj|CPPTRAJ|PMEMD|pmemd|Amber)\D{0,20}(Version|version|V)?\s*[:=]?\s*([0-9][0-9.]*\w*)")


def software(root, out):
    rows = []
    for package in ("numpy", "scipy", "pandas", "matplotlib", "MDAnalysis", "netCDF4", "ParmEd", "scikit-learn",
                    "PyYAML", "snakemake", "snakemake-executor-plugin-slurm", "ovito"):
        try:
            rows.append({"package": package, "version": importlib.metadata.version(package), "source": "python"})
        except importlib.metadata.PackageNotFoundError:
            rows.append({"package": package, "version": "", "source": "not installed"})
    rows.append({"package": "python", "version": sys.version.split()[0], "source": "python"})
    seen = set()
    candidates = sorted(root.glob("results/capillary/*/capwave.log"))[:3]
    m = manifest(root)
    for value in m.get("mdout", []):
        for part in str(value).split(";"):
            if part:
                candidates.append(Path(part))
                break
        if len(candidates) > 6:
            break
    for path in candidates:
        try:
            head = Path(path).read_text(errors="replace")[:4000]
        except OSError:
            continue
        for match in VERSION.finditer(head):
            key = (match.group(1).lower(), match.group(3))
            if key not in seen:
                seen.add(key)
                rows.append({"package": match.group(1), "version": match.group(3), "source": str(path)})
    pd.DataFrame(rows).to_csv(folder(out, "tab:si_software") / "software_versions.csv", index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("."), help="repository root (contains results/ and config/)")
    parser.add_argument("--out", type=Path, default=None, help="default: <root>/results/si_tables")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    out = (args.out or root / "results" / "si_tables").resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    found, missing = copy_sources(root, out)
    inventory(root, out)
    found["tab:si_inventory"].append("(derived) inventory.csv")
    if qstar_rows(root, out):
        found["tab:si_qstar"].append("(derived) qstar_rows.csv")
    for key in endpoint_tables(root, out):
        label = "tab:si_chill" if key == "band" else "tab:si_bragg"
        found[label].append(f"(derived) {key}")
    if not found["tab:si_chill"]:
        missing["tab:si_chill"].append("results/endpoint_structure/<row_id>/local_order/band_order_summary.csv")
    if not found["tab:si_bragg"]:
        missing["tab:si_bragg"].append("results/endpoint_structure/<row_id>/local_order/water_bragg_*.csv")
    software(root, out)
    found["tab:si_software"].append("(derived) software_versions.csv")
    lines = ["# SI table sources", "", "Folder per SI table label. Generated by analysis/si_tables/collect_si_tables.py.", ""]
    for label in SOURCES:
        lines.append(f"## {label}")
        lines += [f"- found: `{p}`" for p in found[label]] or ["- found: nothing"]
        lines += [f"- MISSING: `{p}`" for p in missing[label]]
        lines.append("")
    (out / "INDEX.md").write_text("\n".join(lines))
    n_missing = sum(len(v) for v in missing.values())
    print(f"Wrote {out}; {n_missing} expected source patterns missing (see INDEX.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
