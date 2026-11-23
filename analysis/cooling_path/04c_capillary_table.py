#!/usr/bin/env python3
"""Reduce the cpptraj surftension results of the holds (SI tab:si_capillary).

For every hold row of the manifest the script reads, from
``results/capillary/<row_id>/``:

* ``capwave_summary.dat``   key/value lines written by ``summaryout``
  (``gamma`` is the q^2 S(q) plateau, gamma_app; ``roughness`` is the mean
  roughness of the two interfaces, with ``roughness_upper``/``_lower``);
* ``capwave_spectrum.dat``  S(q) and gamma(q) of both surfaces;
  gamma_0 and kappa' are fitted here as gamma(q) = gamma_0 + kappa' q^2 over
  the summary's q range, the same window cpptraj uses;
* ``capwave.log``           cpptraj stdout, for its own Helfrich intercept,
  which ``summaryout`` does not write (kept as a cross-check);
* ``capwave_blocks.dat``    per-block gamma and roughness, used for the
  drift test that decides the italic temperatures.

Outputs in ``--out`` (default ``results/capillary_summary``):

* ``capillary_by_replica.csv``      one line per hold row;
* ``capillary_by_temperature.csv``  replica mean and SD per temperature,
  with w_CWT and the italic flag;
* ``capillary_spectrum_by_temperature.csv``  replica-mean S(q) and gamma(q)
  per temperature (read by the SI_capillary panel);
* ``capillary_table_rows.tex``      rows to paste into tab:si_capillary.

Drift test. For each replica the block values of gamma and of the mean
roughness are regressed on block index. A replica drifts if either slope
differs from zero at p < --alpha. A temperature is italic when at least
--min-drifting replicas drift. With five blocks this is a coarse screen; the
CSV keeps the slopes and p values.

Example
-------
python analysis/cooling_path/04c_capillary_table.py \\
    --manifest config/manifest.tsv --results results \\
    --out results/capillary_summary
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import ensure_analysis_on_path

ensure_analysis_on_path(__file__)
from opcfilm.io.cpptraj import helfrich_fit, read_helfrich_intercept, read_sets, read_summary

REFERENCE_T = 250.0


def trend(values):
    """OLS slope per block and its two-sided p value; NaN if fewer than three blocks."""
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 3 or np.ptp(y) == 0:
        return (0.0, 1.0) if y.size >= 3 else (math.nan, math.nan)
    fit = stats.linregress(np.arange(y.size), y)
    return float(fit.slope), float(fit.pvalue)


def mean_sd(series):
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return math.nan, math.nan
    return values.mean(), (values.std(ddof=1) if values.size > 1 else math.nan)


def fmt(mean, sd, digits):
    if not np.isfinite(mean):
        return r"\ph"
    if not np.isfinite(sd):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {sd:.{digits}f}"


def one_row(row, directory):
    """Every number of one hold."""
    summary = read_summary(directory / "capwave_summary.dat")
    temperature = float(row["temperature_K"])
    record = {
        "row_id": row["row_id"], "replica": row["replica"], "temperature_K": temperature,
        "frames": summary.get("frames", math.nan),
        "area_A2": summary.get("area", math.nan),
        "qmin_Ainv": summary.get("qmin", math.nan), "qmax_Ainv": summary.get("qmax", math.nan),
        "gamma_app_mN_m": summary.get("gamma", math.nan),
        "gamma_upper_mN_m": summary.get("gamma_upper", math.nan),
        "gamma_lower_mN_m": summary.get("gamma_lower", math.nan),
        "kappa_kT": summary.get("kappa", math.nan),
        "w_A": summary.get("roughness", math.nan),
        "w_upper_A": summary.get("roughness_upper", math.nan),
        "w_lower_A": summary.get("roughness_lower", math.nan),
        "block_mean_gamma_mN_m": summary.get("block_mean_gamma", math.nan),
        "block_sd_gamma_mN_m": summary.get("block_sd_gamma", math.nan),
        "gamma0_log_mN_m": read_helfrich_intercept(directory / "capwave.log"),
    }
    record["gamma_app_over_T"] = record["gamma_app_mN_m"] / temperature
    spectrum = None
    path = directory / "capwave_spectrum.dat"
    if path.is_file():
        spectrum = read_sets(path)
        if "gammaq" in spectrum:
            qmin = record["qmin_Ainv"] if np.isfinite(record["qmin_Ainv"]) else 0.0
            qmax = record["qmax_Ainv"] if np.isfinite(record["qmax_Ainv"]) else 0.30
            g0, kp, n = helfrich_fit(spectrum["x"], spectrum["gammaq"], qmin, qmax)
            record.update(gamma0_mN_m=g0, kappa_prime_mN_m_A2=kp, n_fit_shells=n)
    if not np.isfinite(record.get("gamma0_mN_m", math.nan)):
        record["gamma0_mN_m"] = record["gamma0_log_mN_m"]
    blocks_path = directory / "capwave_blocks.dat"
    blocks = read_sets(blocks_path) if blocks_path.is_file() else pd.DataFrame()
    record["n_blocks"] = len(blocks)
    for aspect, label in (("bgamma", "gamma"), ("bwmean", "w")):
        slope, p = trend(blocks[aspect]) if aspect in blocks else (math.nan, math.nan)
        record[f"{label}_block_slope"] = slope
        record[f"{label}_block_p"] = p
    return record, spectrum


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=Path("config/manifest.tsv"))
    parser.add_argument("--results", type=Path, default=Path("results"),
                        help="the results/ directory (contains capillary/<row_id>/)")
    parser.add_argument("--out", type=Path, default=Path("results/capillary_summary"))
    parser.add_argument("--rate", default="4.44", help="source rate of the holds to tabulate")
    parser.add_argument("--alpha", type=float, default=0.05, help="p threshold of the block drift test")
    parser.add_argument("--min-drifting", type=int, default=2,
                        help="replicas that must drift for a temperature to be italic")
    args = parser.parse_args(argv)

    manifest = pd.read_csv(args.manifest, sep="\t", dtype=str, keep_default_na=False)
    holds = manifest[manifest["kind"] == "hold"]
    column = "source_rate_K_per_ns" if "source_rate_K_per_ns" in holds else "rate_K_per_ns"
    holds = holds[pd.to_numeric(holds[column], errors="coerce") == float(args.rate)]
    if holds.empty:
        sys.exit(f"no hold rows with {column} = {args.rate} in {args.manifest}")

    records, spectra, missing = [], [], []
    for _, row in holds.iterrows():
        directory = args.results / "capillary" / row["row_id"]
        if not (directory / "capwave_summary.dat").is_file():
            missing.append(str(directory / "capwave_summary.dat"))
            continue
        record, spectrum = one_row(row, directory)
        record["drifts"] = bool(any(np.isfinite(p) and p < args.alpha
                                    for p in (record["gamma_block_p"], record["w_block_p"])))
        records.append(record)
        if spectrum is not None:
            spectra.append(spectrum.assign(replica=row["replica"], temperature_K=record["temperature_K"]))
    if missing:
        print(f"warning: {len(missing)} hold rows have no capwave_summary.dat:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
    if not records:
        sys.exit("nothing to tabulate")

    args.out.mkdir(parents=True, exist_ok=True)
    by_replica = pd.DataFrame(records).sort_values(["temperature_K", "replica"], ascending=[False, True])
    by_replica.to_csv(args.out / "capillary_by_replica.csv", index=False)

    rows = []
    for temperature, group in by_replica.groupby("temperature_K", sort=False):
        entry = {"temperature_K": temperature, "n_replicas": len(group), "n_drifting": int(group["drifts"].sum())}
        for name in ("gamma_app_mN_m", "gamma0_mN_m", "gamma0_log_mN_m", "kappa_prime_mN_m_A2",
                     "gamma_app_over_T", "w_A", "w_upper_A", "w_lower_A", "kappa_kT"):
            if name in group:
                entry[f"{name}_mean"], entry[f"{name}_sd"] = mean_sd(group[name])
        rows.append(entry)
    by_t = pd.DataFrame(rows).sort_values("temperature_K", ascending=False)
    by_t["italic"] = by_t["n_drifting"] >= args.min_drifting
    reference = by_t[np.isclose(by_t["temperature_K"], REFERENCE_T)]
    if reference.empty:
        print(f"warning: no {REFERENCE_T:g}-K hold; w_CWT left empty", file=sys.stderr)
        by_t["w_CWT_A"] = math.nan
    else:
        w_ref = reference["w_A_mean"].iloc[0]
        g_ref = reference["gamma0_mN_m_mean"].iloc[0]
        # w_CWT = w(250 K) [(T / gamma_0(T)) / (250 K / gamma_0(250 K))]^(1/2)
        by_t["w_CWT_A"] = w_ref * np.sqrt((by_t["temperature_K"] / by_t["gamma0_mN_m_mean"]) / (REFERENCE_T / g_ref))
    by_t.to_csv(args.out / "capillary_by_temperature.csv", index=False)

    if spectra:
        stacked = pd.concat(spectra, ignore_index=True)
        value_columns = [c for c in stacked.columns if c not in ("x", "replica", "temperature_K")]
        grouped = stacked.groupby(["temperature_K", "x"])[value_columns]
        mean = grouped.mean().add_suffix("_mean")
        sd = grouped.std().add_suffix("_sd")
        table = mean.join(sd).reset_index().rename(columns={"x": "q_Ainv"})
        table["n_replicas"] = stacked.groupby(["temperature_K", "x"])["replica"].nunique().to_numpy()
        table.to_csv(args.out / "capillary_spectrum_by_temperature.csv", index=False)

    lines = []
    for _, r in by_t.iterrows():
        label = f"{r['temperature_K']:g}"
        if r["italic"]:
            label = f"\\textit{{{label}}}"
        w_cwt = "---" if np.isclose(r["temperature_K"], REFERENCE_T) else fmt(r["w_CWT_A"], math.nan, 2)
        cells = [label,
                 fmt(r["gamma_app_mN_m_mean"], r["gamma_app_mN_m_sd"], 1),
                 fmt(r["gamma0_mN_m_mean"], r["gamma0_mN_m_sd"], 1),
                 fmt(r["gamma_app_over_T_mean"], r["gamma_app_over_T_sd"], 3),
                 fmt(r["w_A_mean"], r["w_A_sd"], 2),
                 w_cwt]
        lines.append(" & ".join(cells) + r" \\")
    tex = "\n".join(lines) + "\n"
    (args.out / "capillary_table_rows.tex").write_text(tex)
    print(tex, end="")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
