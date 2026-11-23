#!/usr/bin/env python3
"""Tabulate the cpptraj surftension results of the holds (SI Table tab:si_capillary).

For every hold row of the manifest the script reads, from
``results/capillary/<row_id>/``:

* ``capwave_summary.dat``  key/value lines written by ``summaryout``
  (``gamma`` is the q^2 S(q) plateau, i.e. gamma_app; ``roughness`` is the
  mean roughness of the two interfaces);
* ``capwave.log``          cpptraj stdout, for the Helfrich intercept gamma_0
  (``gamma, Helfrich intercept = ... mN/m``), which summaryout does not write;
* ``capwave_blocks.dat``   per-block gamma and roughness (``bgamma``, ``bwmean``),
  used for the drift test that decides the italic temperatures.

Outputs (in --out):

* ``capillary_by_replica.csv``   one line per hold row;
* ``capillary_by_temperature.csv`` replica mean and SD per temperature;
* ``capillary_table_rows.tex``   rows to paste between \\midrule and
  \\bottomrule of tab:si_capillary.

Drift test. For each replica the block values of gamma and of the mean
roughness are regressed on block index (ordinary least squares). A replica
drifts if either slope differs from zero at p < --alpha. A temperature is set
in italics when at least --min-drifting replicas drift. With five blocks this
is a coarse screen; the CSV keeps the slopes and p values so the call can be
made by eye.

Example
-------
python analysis/cooling_path/04c_capillary_table.py \\
    --manifest config/manifest.tsv --results results \\
    --out results/capillary_summary
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HELFRICH = re.compile(r"gamma,\s*Helfrich intercept\s*=\s*([-+0-9.eEnaNinf]+)\s*mN/m")
REFERENCE_T = 250.0


def read_summary(path):
    """Key/value pairs of a surftension summaryout file."""
    values = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        key, value = parts
        try:
            values[key] = float(value)
        except ValueError:
            values[key] = value
    return values


def read_helfrich(path):
    """gamma_0 from the cpptraj log, or NaN if the fit was not printed."""
    if not path.is_file():
        return math.nan
    found = HELFRICH.findall(path.read_text(errors="replace"))
    return float(found[-1]) if found else math.nan


def read_blocks(path):
    """Block table with columns named by their cpptraj aspect (bgamma, bwmean, ...)."""
    lines = path.read_text().splitlines()
    header = next((l for l in lines if l.startswith("#")), None)
    rows = [l.split() for l in lines if l.strip() and not l.startswith("#")]
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows).apply(pd.to_numeric, errors="coerce")
    if header is not None:
        names = header.lstrip("#").split()
        if len(names) == table.shape[1]:
            # "ST_00001[bgamma]" -> "bgamma"; the first column is the block index.
            table.columns = [re.sub(r"^.*\[(.*)\]$", r"\1", n) for n in names]
    return table


def trend(values):
    """OLS slope per block and its two-sided p value; NaN if fewer than three blocks."""
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 3:
        return math.nan, math.nan
    fit = stats.linregress(np.arange(y.size), y)
    return fit.slope, fit.pvalue


def mean_sd(series):
    values = series.dropna().to_numpy(dtype=float)
    if values.size == 0:
        return math.nan, math.nan, 0
    sd = values.std(ddof=1) if values.size > 1 else math.nan
    return values.mean(), sd, values.size


def fmt(mean, sd, digits):
    if not np.isfinite(mean):
        return r"\ph"
    if not np.isfinite(sd):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {sd:.{digits}f}"


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
    rate_column = "source_rate_K_per_ns" if "source_rate_K_per_ns" in holds else "rate_K_per_ns"
    holds = holds[holds[rate_column].astype(float) == float(args.rate)]
    if holds.empty:
        sys.exit(f"no hold rows with {rate_column} = {args.rate} in {args.manifest}")

    records, missing = [], []
    for _, row in holds.iterrows():
        directory = args.results / "capillary" / row["row_id"]
        summary_path = directory / "capwave_summary.dat"
        if not summary_path.is_file():
            missing.append(str(summary_path))
            continue
        summary = read_summary(summary_path)
        temperature = float(row["temperature_K"])
        record = {
            "row_id": row["row_id"],
            "replica": row["replica"],
            "temperature_K": temperature,
            "frames": summary.get("frames", math.nan),
            "gamma_app_mN_m": summary.get("gamma", math.nan),
            "gamma_upper_mN_m": summary.get("gamma_upper", math.nan),
            "gamma_lower_mN_m": summary.get("gamma_lower", math.nan),
            "gamma0_mN_m": read_helfrich(directory / "capwave.log"),
            "kappa_kT": summary.get("kappa", math.nan),
            "w_A": summary.get("roughness", math.nan),
            "w_upper_A": summary.get("roughness_upper", math.nan),
            "w_lower_A": summary.get("roughness_lower", math.nan),
            "block_mean_gamma": summary.get("block_mean_gamma", math.nan),
            "block_sd_gamma": summary.get("block_sd_gamma", math.nan),
        }
        record["gamma_app_over_T"] = record["gamma_app_mN_m"] / temperature
        blocks_path = directory / "capwave_blocks.dat"
        blocks = read_blocks(blocks_path) if blocks_path.is_file() else pd.DataFrame()
        record["n_blocks"] = len(blocks)
        for aspect, label in (("bgamma", "gamma"), ("bwmean", "w")):
            if aspect in blocks:
                slope, p = trend(blocks[aspect])
            else:
                slope, p = math.nan, math.nan
            record[f"{label}_block_slope"] = slope
            record[f"{label}_block_p"] = p
        record["drifts"] = bool(
            any(np.isfinite(p) and p < args.alpha for p in (record["gamma_block_p"], record["w_block_p"]))
        )
        if not np.isfinite(record["gamma0_mN_m"]):
            print(f"warning: no Helfrich intercept in {directory / 'capwave.log'}", file=sys.stderr)
        if blocks.empty:
            print(f"warning: no block table for {row['row_id']}", file=sys.stderr)
        records.append(record)

    if missing:
        print(f"warning: {len(missing)} hold rows have no capwave_summary.dat:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
    if not records:
        sys.exit("nothing to tabulate")

    by_replica = pd.DataFrame(records).sort_values(["temperature_K", "replica"], ascending=[False, True])
    args.out.mkdir(parents=True, exist_ok=True)
    by_replica.to_csv(args.out / "capillary_by_replica.csv", index=False)

    summary_rows = []
    for temperature, group in by_replica.groupby("temperature_K", sort=False):
        entry = {"temperature_K": temperature, "n_replicas": len(group),
                 "n_drifting": int(group["drifts"].sum())}
        for column in ("gamma_app_mN_m", "gamma0_mN_m", "gamma_app_over_T", "w_A", "kappa_kT"):
            entry[f"{column}_mean"], entry[f"{column}_sd"], _ = mean_sd(group[column])
        summary_rows.append(entry)
    by_temperature = pd.DataFrame(summary_rows).sort_values("temperature_K", ascending=False)
    by_temperature["italic"] = by_temperature["n_drifting"] >= args.min_drifting

    reference = by_temperature[by_temperature["temperature_K"] == REFERENCE_T]
    if reference.empty:
        print(f"warning: no {REFERENCE_T:g}-K hold; w_CWT left empty", file=sys.stderr)
        by_temperature["w_CWT_A"] = math.nan
    else:
        w_ref = reference["w_A_mean"].iloc[0]
        g_ref = reference["gamma0_mN_m_mean"].iloc[0]
        # w_CWT = w(250 K) [(T / gamma_0(T)) / (250 K / gamma_0(250 K))]^(1/2)
        by_temperature["w_CWT_A"] = w_ref * np.sqrt(
            (by_temperature["temperature_K"] / by_temperature["gamma0_mN_m_mean"]) / (REFERENCE_T / g_ref)
        )
    by_temperature.to_csv(args.out / "capillary_by_temperature.csv", index=False)

    lines = []
    for _, r in by_temperature.iterrows():
        label = f"{r['temperature_K']:g}"
        if r["italic"]:
            label = f"\\textit{{{label}}}"
        w_cwt = "---" if r["temperature_K"] == REFERENCE_T else fmt(r["w_CWT_A"], math.nan, 2)
        cells = [
            label,
            fmt(r["gamma_app_mN_m_mean"], r["gamma_app_mN_m_sd"], 1),
            fmt(r["gamma0_mN_m_mean"], r["gamma0_mN_m_sd"], 1),
            fmt(r["gamma_app_over_T_mean"], r["gamma_app_over_T_sd"], 3),
            fmt(r["w_A_mean"], r["w_A_sd"], 2),
            w_cwt,
        ]
        lines.append(" & ".join(cells) + r" \\")
    tex = "\n".join(lines) + "\n"
    (args.out / "capillary_table_rows.tex").write_text(tex)
    print(tex, end="")
    print(f"wrote {args.out}/capillary_by_replica.csv, capillary_by_temperature.csv, capillary_table_rows.tex",
          file=sys.stderr)


if __name__ == "__main__":
    main()
