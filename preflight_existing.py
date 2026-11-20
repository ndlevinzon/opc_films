#!/usr/bin/env python3
"""Check a manifest built from trajectories you have already run, before any analysis job is submitted.

Usage (from the repository root, after `opcfilm catalog build`):

    python preflight_existing.py --manifest config/manifest.tsv            # fast: manifest and mdout headers only
    python preflight_existing.py --manifest config/manifest.tsv --deep     # also opens topology and every NetCDF

It exits non-zero if any FAIL line is printed. WARN lines need a decision, not
necessarily a fix. Nothing is written.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from opcfilm.catalog import RATE_CODES, REPLICAS, as_physical_rate, load_manifest, trajectory_list  # noqa: E402
from opcfilm.config import CONFIG  # noqa: E402

TEMP0 = re.compile(r"(?m)^\s*temp0\s*=\s*([0-9]*\.?[0-9]+)")
NTWX = re.compile(r"(?m)^\s*ntwx\s*=\s*([0-9]+)")
COOL_RST = re.compile(r"cool_(\d+)ns_chunk_(\d+)_(\d{3})_to_(\d{3})K\.rst7$")

fails, warns = [], []


def fail(msg):
    fails.append(msg)
    print("FAIL ", msg)


def warn(msg):
    warns.append(msg)
    print("WARN ", msg)


def header(path, nbytes=200_000):
    try:
        with open(path, "r", errors="replace") as handle:
            return handle.read(nbytes)
    except OSError as exc:
        return f"<unreadable: {exc}>"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="config/manifest.tsv")
    ap.add_argument("--deep", action="store_true", help="Open the topology and every NetCDF (needs parmed and netCDF4).")
    ap.add_argument("--min-hold-ns", type=float, default=None, help="Warn when a hold is shorter than this.")
    args = ap.parse_args()

    rows = load_manifest(args.manifest)
    if not rows:
        fail(f"{args.manifest} has no rows. Check --root and the directory layout (section 2 of the readiness note).")
        return finish()

    by_kind = Counter(r["kind"] for r in rows)
    print(f"Rows: {dict(by_kind)}")

    # 1. Coverage --------------------------------------------------------------
    ramps = defaultdict(set)
    for r in rows:
        if r["kind"] == "ramp":
            ramps[(r["replica"], r["rate_K_per_ns"])].add(r["temperature_K"])
    expected_rates = {f"{as_physical_rate(c):g}" for c in RATE_CODES}
    for rep in REPLICAS:
        for rate in sorted(expected_rates, key=float):
            n = len(ramps.get((rep, rate), ()))
            if n != 23:
                (fail if n == 0 else warn)(f"ramp {rep} {rate} K/ns has {n} windows, expected 23 (300->80 K in 10 K steps, then 80->78 K).")
    holds = defaultdict(set)
    for r in rows:
        if r["kind"] == "hold":
            holds[r["replica"]].add(float(r["temperature_K"] or "nan"))
    sets = {rep: tuple(sorted(ts)) for rep, ts in holds.items()}
    if not holds:
        fail("No hold rows. Holds must be at opc/<rep>/hold/<T>K/TRAJ/traj.N with OUT/mdout.N (real directories, not symlinked directories).")
    elif len(set(sets.values())) != 1 or set(sets) != set(REPLICAS):
        fail(f"Hold temperatures differ between replicas or a replica is missing: {sets}")
    else:
        print(f"Hold temperatures (all replicas): {next(iter(sets.values()))}")
    ends = Counter((r["replica"], r["temperature_K"]) for r in rows if r["kind"] == "endpoint")
    for rep in REPLICAS:
        for t in ("300", "78"):
            if ends.get((rep, t), 0) != 1:
                fail(f"endpoint {rep} {t} K: {ends.get((rep, t), 0)} rows, expected exactly 1 per replica (a concatenated cross-replica file cannot train the classifier).")

    # 2. Per-row metadata -------------------------------------------------------
    cuts = defaultdict(Counter)
    for r in rows:
        rid = r.get("row_id") or r.get("label")
        for col in ("mdout", "cut_A", "duration_ps", "frame_ps", "topology", "files"):
            if not str(r.get(col, "")).strip():
                fail(f"{rid}: blank {col}.")
        if r.get("cut_A"):
            cuts[r["kind"]][float(r["cut_A"])] += 1
        for path in trajectory_list(r):
            p = Path(path)
            if not p.is_file() or p.stat().st_size == 0:
                fail(f"{rid}: trajectory missing or empty: {p}")
        mdouts = [m for m in str(r.get("mdout", "")).split(";") if m]
        if r["kind"] in ("hold", "endpoint"):
            setpoint = float(r["temperature_K"])
            for m in mdouts:
                text = header(m)
                t0 = TEMP0.search(text)
                if not t0:
                    warn(f"{rid}: no temp0 in {m}")
                elif abs(float(t0.group(1)) - setpoint) > 0.5:
                    fail(f"{rid}: {Path(m).name} has temp0 = {t0.group(1)}, setpoint is {setpoint:g} K (archived prop_trajectory.mdin used 300 K).")
                nx = NTWX.search(text)
                if nx and int(nx.group(1)) != int(CONFIG.trajectory.hold_ntwx):
                    warn(f"{rid}: {Path(m).name} ntwx = {nx.group(1)}, sheet says {CONFIG.trajectory.hold_ntwx}.")
            restart = Path(str(r.get("source_restart", ""))).name
            mt = COOL_RST.search(restart)
            if r["kind"] == "hold" or setpoint == 78:
                if not mt:
                    fail(f"{rid}: first segment started from '{restart or '?'}', not a cooling restart. "
                         "A hold started from min/step9.rst7 is a quench from 300 K, not a continuation of the ramp.")
                elif int(mt.group(4)) != int(round(setpoint)):
                    fail(f"{rid}: started from {restart}, which ends at {int(mt.group(4))} K, not {setpoint:g} K.")
                if not r.get("source_rate_K_per_ns"):
                    fail(f"{rid}: blank source_rate_K_per_ns; tg_dyn.sh would silently leave this hold out of T_g.")
            elif setpoint == 300:
                print(f"INFO  {rid}: 300 K endpoint starts from '{restart or '?'}'. Record this in the methods.")
            dur = float(r.get("duration_ps") or 0)
            fps = float(r.get("frame_ps") or 0)
            if fps > 0:
                kept = int(dur * (1 - CONFIG.discard.hold_fraction) / fps)
                if r["kind"] == "endpoint" and kept < CONFIG.classifier.frames:
                    fail(f"{rid}: only {kept} frames after the 10% discard; the classifier needs {CONFIG.classifier.frames}.")
            if args.min_hold_ns and r["kind"] == "hold" and dur < args.min_hold_ns * 1000:
                warn(f"{rid}: hold is {dur / 1000:g} ns (< {args.min_hold_ns:g} ns).")

    # 3. Cutoff consistency ------------------------------------------------------
    print("cut_A by kind:", {k: dict(v) for k, v in cuts.items()})
    allcuts = {c for v in cuts.values() for c in v}
    if len(allcuts) > 1:
        fail(f"Mixed real-space cutoffs {sorted(allcuts)} Å. Holds/endpoints and ramps must match, or the methods must say so and you must decide (re-run holds at 9 Å, or accept and document).")
    elif allcuts and allcuts != {float(CONFIG.trajectory.production_cut_A)}:
        warn(f"All runs used cut = {allcuts}, sheet says {CONFIG.trajectory.production_cut_A}. Update project.yaml and the methods.")

    # 4. Deep checks ---------------------------------------------------------------
    if args.deep:
        from opcfilm.io.trajectory import count_waters, netcdf_time_ps
        tops = {r["topology"] for r in rows}
        for t in tops:
            n = count_waters(t)
            if n != CONFIG.film.n_water:
                fail(f"{t}: {n} waters, sheet says {CONFIG.film.n_water}.")
        for r in rows:
            rid = r.get("row_id")
            total = 0
            for path in trajectory_list(r):
                try:
                    times = netcdf_time_ps(path)
                except Exception as exc:  # truncated or unreadable file
                    fail(f"{rid}: cannot read {path}: {exc}")
                    continue
                total += len(times)
            fps = float(r.get("frame_ps") or 0)
            dur = float(r.get("duration_ps") or 0)
            if fps and dur and abs(total - dur / fps) > max(2, 0.02 * dur / fps):
                fail(f"{rid}: {total} frames on disk, mdout implies {dur / fps:.0f} (truncated or missing segment?).")
    return finish()


def finish():
    print(f"\n{len(fails)} FAIL, {len(warns)} WARN")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
