"""Command line for the catalog, one stage, and the synthetic film.

``opcfilm catalog build --root`` writes the manifest. ``opcfilm s00 --row 1``
runs one manifest row. ``opcfilm reduce s00`` concatenates that stage.
``opcfilm synthetic`` runs every stage on a 2,000-water film.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from opcfilm.catalog import (
    COLUMNS,
    REPLICAS,
    RATE_CODES,
    cutoff_note,
    discover_rows,
    load_manifest,
    result_directory,
    select_row,
    select_row_id,
    write_manifest,
)
from opcfilm.metadata import require_clean_worktree
from opcfilm.stages import GROUP, REDUCTIONS, SCRIPTS, run as run_stage
from opcfilm.stages.reduce import reduce_stage, summarize_stage


def _repo():
    for parent in Path(__file__).resolve().parents:
        if (parent / "config" / "project.yaml").is_file():
            return parent
    return Path.cwd()


def _build(args):
    if not str(args.root).strip():
        raise SystemExit("Set OPC_DATA_ROOT or pass --root. No data root was given.")
    root = Path(args.root)
    topology = Path(args.topology) if args.topology else root / "build" / "opc_slab.prmtop"
    workflow = Path(args.workflow) if args.workflow else _repo()
    rows = discover_rows(root, workflow, topology, REPLICAS, RATE_CODES)
    destination = Path(args.out)
    if not rows:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("\t".join(COLUMNS) + "\n", encoding="utf-8")
        print(f"No trajectories under {root}. Wrote the header to {destination}.")
        return 0
    write_manifest(destination, rows)
    print(cutoff_note(rows))
    print(f"Wrote {len(rows)} rows to {destination}.")
    return 0


def _stage(name, args):
    if name in REDUCTIONS or name in GROUP:
        run_stage(name, None, args.root)
        return 0
    rows = load_manifest(args.manifest)
    if args.row_id:
        row = select_row_id(rows, args.row_id)
    elif args.row is not None:
        row = select_row(rows, args.row)
    else:
        raise SystemExit("Pass --row-id. --row is the 1-based line number.")
    row["stage"] = name
    row["outdir"] = str(result_directory(args.root, name, row))
    run_stage(name, row, args.root)
    return 0


def main(argv=None):
    """Run a catalog, stage, reduction, or the synthetic film.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments, without the program name. The default reads ``sys.argv``.

    Returns
    -------
    int
        Zero when the command finished.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Run outside a clean git checkout.",
    )
    common.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="section.field=value",
        help="Sensitivity override. It is recorded in run_metadata.json.",
    )
    parser = argparse.ArgumentParser(prog="opcfilm", parents=[common])
    sub = parser.add_subparsers(dest="command", required=True)

    catalog = sub.add_parser("catalog", parents=[common])
    catalog_sub = catalog.add_subparsers(dest="catalog_command", required=True)
    build = catalog_sub.add_parser("build", parents=[common])
    build.add_argument("--root", required=True)
    build.add_argument("--topology", default=None)
    build.add_argument("--workflow", default=None)
    build.add_argument("--out", default=str(_repo() / "config" / "manifest.tsv"))

    for name in SCRIPTS:
        stage = sub.add_parser(name, parents=[common])
        stage.add_argument("--row", type=int, default=None)
        stage.add_argument("--row-id", default=None)
        stage.add_argument("--manifest", default=str(_repo() / "config" / "manifest.tsv"))
        stage.add_argument("--root", default=str(_repo()))

    reduce = sub.add_parser("reduce", parents=[common])
    reduce.add_argument("stage", choices=tuple(SCRIPTS))
    reduce.add_argument("--root", default=str(_repo()))
    reduce.add_argument("--summarize", action="store_true")
    reduce.add_argument("--allow-incomplete", action="store_true")

    synthetic = sub.add_parser("synthetic", parents=[common])
    synthetic.add_argument("--root", default=None)

    md = sub.add_parser("md", parents=[common])
    md_sub = md.add_subparsers(dest="md_command", required=True)
    render_cmd = md_sub.add_parser("render", parents=[common])
    render_cmd.add_argument("--kind", choices=("cool", "hold"), required=True)
    render_cmd.add_argument("--temperature", type=float, required=True)
    render_cmd.add_argument("--temperature-end", type=float, default=None)
    render_cmd.add_argument("--length-ns", type=float, required=True)
    render_cmd.add_argument("--out", required=True)

    qstar = sub.add_parser("qstar", parents=[common])
    qstar_sub = qstar.add_subparsers(dest="qstar_command", required=True)
    adopt = qstar_sub.add_parser("adopt", parents=[common])
    adopt.add_argument("paths", nargs="*", help="Per-row qstar_row.json files.")
    adopt.add_argument("--manifest", default=None)
    adopt.add_argument("--out", default=str(_repo() / "outputs" / "qstar.json"))

    args = parser.parse_args(argv)
    if args.set:
        from opcfilm.config import apply_cli_overrides

        apply_cli_overrides(args.set)
    if args.command != "synthetic":
        require_clean_worktree(_repo(), allow_dirty=args.allow_dirty)
    if args.command == "catalog":
        return _build(args)
    if args.command == "reduce":
        if args.summarize:
            replicas, contrast = summarize_stage(
                args.stage, args.root, allow_incomplete=args.allow_incomplete
            )
            print(replicas)
            print(contrast)
        else:
            print(reduce_stage(args.stage, args.root))
        return 0
    if args.command == "md":
        from opcfilm.mdinput import render

        path = render(
            args.kind,
            args.temperature,
            args.length_ns,
            args.out,
            args.temperature_end,
        )
        print(path)
        return 0
    if args.command == "synthetic":
        from opcfilm.synthetic import run

        run(Path(args.root) if args.root else None)
        return 0
    if args.command == "qstar":
        from opcfilm.qstar import adopt, adopt_files

        if args.paths:
            payload = adopt_files(args.paths, args.out)
        else:
            manifest = args.manifest or str(_repo() / "config" / "manifest.tsv")
            payload = adopt(manifest, args.out)
        print(f"Adopted q* = {payload['qstar_adopted_Ainv']:.6f} from {payload['n_rows']} rows.")
        print(args.out)
        return 0
    return _stage(args.command, args)


if __name__ == "__main__":
    sys.exit(main())
