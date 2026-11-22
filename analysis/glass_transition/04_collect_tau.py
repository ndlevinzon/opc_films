#!/usr/bin/env python3

import argparse
import glob
import os

import numpy as np
import pandas as pd


def main():

    """Combine per-temperature relaxation tables into one CSV.

    Returns
    -------
    result
        Combine per-temperature relaxation tables into one CSV.
    """
    ap = argparse.ArgumentParser(
        description=(
            "Collect Stage-3 alpha-relaxation summaries "
            "into master tables."
        )
    )

    ap.add_argument(
        "patterns",
        nargs="+",
        help="Glob patterns for tau_T*_r*.csv files"
    )

    ap.add_argument(
        "-o", "--out",
        required=True,
        help="Output CSV containing all whole/block rows"
    )

    ap.add_argument(
        "--classified",
        default=None,
        help=(
            "Classified CSV from 03b_plot_relaxation.py. "
            "If omitted, tau_status from 03_fs_alpha.py is copied to status."
        )
    )

    args = ap.parse_args()


    # ============================================================
    # Find input files
    # ============================================================

    files = []

    for pattern in args.patterns:
        files.extend(glob.glob(pattern))

    files = sorted(set(files))

    if not files:
        raise SystemExit("No summary CSVs found")


    print(f"Found {len(files)} summary files.")


    # ============================================================
    # Read and combine
    # ============================================================

    frames = []

    for f in files:

        d = pd.read_csv(f)

        d["source_file"] = os.path.relpath(f)

        frames.append(d)

    df = pd.concat(
        frames,
        ignore_index=True
    )


    # ============================================================
    # Required columns
    # ============================================================

    required = [
        "block",
        "T_K",
        "replica",
        "qstar_Ainv",
        "tau_1e_ns"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing required columns: "
            + ", ".join(missing)
        )


    # ============================================================
    # Normalize data types
    # ============================================================

    df["T_K"] = pd.to_numeric(
        df["T_K"],
        errors="coerce"
    )

    # Replicas are labels such as "r1" from the manifest. Converting them to
    # numbers turned every label into NaN and merged all replicas.
    df["replica"] = df["replica"].astype(str).str.strip()
    unlabeled = df["replica"].isin(["", "nan", "NA", "None"])
    if unlabeled.any():
        raise RuntimeError(
            "Summaries without a replica label: "
            + ", ".join(sorted(df.loc[unlabeled, "source_file"].astype(str).unique()))
        )

    df["tau_1e_ns"] = pd.to_numeric(
        df["tau_1e_ns"],
        errors="coerce"
    )


    # ============================================================
    # Check uniqueness
    #
    # There should be one row per:
    #
    # T / replica / block
    #
    # ============================================================

    key = [
        "T_K",
        "replica",
        "block"
    ]

    duplicate = df.duplicated(
        key,
        keep=False
    )

    if duplicate.any():

        print()
        print("ERROR: duplicate T/replica/block rows:")
        print(
            df.loc[
                duplicate,
                key + ["source_file"]
            ].sort_values(key)
        )

        raise RuntimeError(
            "Duplicate relaxation rows detected."
        )


    # ============================================================
    # Merge Stage-3b censoring classification
    # ============================================================

    if args.classified is not None:

        c = pd.read_csv(
            args.classified
        )

        rename = {
            "temperature_K": "T_K"
        }

        c = c.rename(
            columns=rename
        )

        keep = [
            "T_K",
            "replica",
            "block",
            "status",
            "tau_plot_ns",
            "first_lag_ns",
            "max_lag_ns",
            "first_Fs",
            "min_Fs"
        ]

        keep = [
            x for x in keep
            if x in c.columns
        ]

        c = c[keep].copy()
        if "replica" in c.columns:
            c["replica"] = c["replica"].astype(str).str.strip()

        df = df.merge(
            c,
            on=[
                "T_K",
                "replica",
                "block"
            ],
            how="left",
            validate="one_to_one"
        )

    if "status" not in df.columns and "tau_status" in df.columns:
        df["status"] = df["tau_status"]
        print("Copied tau_status to status.")

    if "status" not in df.columns:
        raise RuntimeError(
            "No status column. Collect 03_fs_alpha.py summaries, which "
            "contain tau_status, or pass --classified from "
            "03b_plot_relaxation.py."
        )


    # ============================================================
    # Derived quantities
    # ============================================================

    df["invT_1000_Kinv"] = (
        1000.0 / df["T_K"]
    )

    df["log10_tau_1e_ns"] = np.nan

    good = (
        np.isfinite(df["tau_1e_ns"])
        &
        (df["tau_1e_ns"] > 0)
    )

    df.loc[
        good,
        "log10_tau_1e_ns"
    ] = np.log10(
        df.loc[
            good,
            "tau_1e_ns"
        ]
    )


    # ============================================================
    # Sort
    # ============================================================

    block_order = {
        "whole": 0,
        "block1": 1,
        "block2": 2,
        "block3": 3
    }

    df["_block_order"] = (
        df["block"]
        .map(block_order)
        .fillna(99)
    )

    df = df.sort_values(
        [
            "T_K",
            "replica",
            "_block_order"
        ],
        ascending=[
            False,
            True,
            True
        ]
    )

    df = df.drop(
        columns="_block_order"
    )


    # ============================================================
    # Write complete table
    # ============================================================

    df.to_csv(
        args.out,
        index=False
    )

    print()
    print(
        f"Wrote {len(df)} rows to:"
    )
    print(
        f"  {args.out}"
    )


    # ============================================================
    # Write WHOLE trajectory table
    #
    # This is the table to use later for tau(T), VFT, etc.
    # ============================================================

    whole = df[
        df["block"] == "whole"
    ].copy()

    root, ext = os.path.splitext(
        args.out
    )

    whole_out = (
        root + "_whole" + ext
    )

    whole.to_csv(
        whole_out,
        index=False
    )

    print(
        f"Wrote {len(whole)} whole-trajectory rows to:"
    )
    print(
        f"  {whole_out}"
    )


    # ============================================================
    # Print useful summary
    # ============================================================

    print()
    print(
        "Whole-trajectory relaxation data:"
    )
    print()

    cols = [
        "T_K",
        "replica",
        "tau_1e_ns"
    ]

    if "status" in whole.columns:
        cols.append("status")

    if "data_source" in whole.columns:
        cols.append("data_source")

    print(
        whole[cols]
        .to_string(index=False)
    )


    # ============================================================
    # Classification counts
    # ============================================================

    if "status" in whole.columns:

        print()
        print(
            "Whole-trajectory status counts:"
        )

        print(
            whole["status"]
            .value_counts(
                dropna=False
            )
        )


if __name__ == "__main__":
    main()
