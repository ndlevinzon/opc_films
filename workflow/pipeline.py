"""Array-job fallback for the Phase A graph.

Snakemake is the runner. This module submits one ``sbatch --array`` job per
stage. Each stage waits for the previous stage's job id. It does not call
``opcfilm s00``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _rows(manifest, kinds, tmin=None, tmax=None):
    """1-based manifest line numbers.

    Parameters
    ----------
    manifest : path-like
        ``config/manifest.tsv``.
    kinds : sequence of str
        Row kinds to keep.
    tmin, tmax : float, optional
        Inclusive temperature limits.

    Returns
    -------
    list of int
        Line numbers starting at 1. An empty manifest returns an empty list.
    """
    frame = pd.read_csv(manifest, sep="\t", dtype=str, keep_default_na=False)
    if frame.empty or "kind" not in frame.columns:
        return []
    selected = frame[frame.kind.isin(list(kinds))].copy()
    if tmin is not None and not selected.empty:
        selected = selected[selected.temperature_K.astype(float) >= tmin]
    if tmax is not None and not selected.empty:
        selected = selected[selected.temperature_K.astype(float) <= tmax]
    return [int(index) + 1 for index in selected.index]


def _array(indices):
    """SLURM array specification.

    Parameters
    ----------
    indices : sequence of int
        1-based manifest rows.

    Returns
    -------
    str
        A comma-separated ``--array`` value.
    """
    return ",".join(str(index) for index in indices)


def submit(name, command, dependency=None, array=None):
    """Submit one job.

    Parameters
    ----------
    name : str
        Job name.
    command : str
        Command passed to ``sbatch --wrap``. ``$SLURM_ARRAY_TASK_ID`` is the
        manifest row when ``array`` is set.
    dependency : str, optional
        Job id that must finish first.
    array : str, optional
        Value of ``--array``.

    Returns
    -------
    str
        Scheduler job id.
    """
    if array == "":
        return ""
    argv = ["sbatch", "--parsable", "--job-name", name]
    if array:
        argv.extend(["--array", array])
    if dependency:
        earlier = [part for part in str(dependency).split(":") if part]
        if earlier:
            argv.extend(["--dependency", "afterok:" + ":".join(earlier)])
    argv.extend(["--wrap", command])
    completed = subprocess.run(argv, check=True, capture_output=True, text=True)
    return completed.stdout.strip().split(";")[0]


def submit_graph(manifest):
    """Submit the Phase A stages in dependency order.

    Parameters
    ----------
    manifest : path-like
        Reviewed manifest.

    Returns
    -------
    list of tuple
        ``(stage, job_id)``.
    """
    manifest = str(manifest)
    ramp = _array(_rows(manifest, ["ramp"]))
    hold = _array(_rows(manifest, ["hold"]))
    qstar = _array(_rows(manifest, ["hold"], 150, 250))
    relaxation = _array(_rows(manifest, ["hold", "endpoint"]))
    both = _array(_rows(manifest, ["ramp", "hold"]))
    submitted = []
    profiles = submit(
        "profiles",
        f"python analysis/cooling_path/03_run_row.py --manifest {manifest} --row $SLURM_ARRAY_TASK_ID",
        array=ramp,
    )
    submitted.append(("profiles", profiles))
    capillary = submit(
        "capillary",
        f"python analysis/cooling_path/03_run_row.py --manifest {manifest} --row $SLURM_ARRAY_TASK_ID --capwave",
        array=hold,
    )
    submitted.append(("capillary", capillary))
    qstar_row = submit(
        "qstar_row",
        "bash workflow/reduce/qstar_row.sh $SLURM_ARRAY_TASK_ID",
        array=qstar,
    )
    submitted.append(("qstar_row", qstar_row))
    adopt = submit(
        "qstar_adopt",
        "opcfilm qstar adopt results/qstar_row/*/qstar_row.json --out outputs/qstar.json",
        dependency=qstar_row,
    )
    submitted.append(("qstar_adopt", adopt))
    fs_tau = submit(
        "fs_tau",
        "bash workflow/reduce/fs_tau_row.sh $SLURM_ARRAY_TASK_ID",
        dependency=adopt,
        array=relaxation,
    )
    submitted.append(("fs_tau", fs_tau))
    tg = submit("tg_dyn", "bash workflow/reduce/tg_dyn.sh", dependency=fs_tau)
    submitted.append(("tg_dyn", tg))
    energy = submit("energy_arrest", "bash workflow/reduce/energy_arrest.sh", dependency=profiles)
    submitted.append(("energy_arrest", energy))
    regional = submit(
        "regional_dyn",
        "bash workflow/reduce/regional_row.sh $SLURM_ARRAY_TASK_ID",
        dependency=adopt,
        array=both,
    )
    submitted.append(("regional_dyn", regional))
    hbonds = submit(
        "hbond",
        f"python analysis/interface_dynamics/run_hb_manifest.py --manifest {manifest} --row $SLURM_ARRAY_TASK_ID",
        dependency=regional,
        array=both,
    )
    submitted.append(("hbond", hbonds))
    defects = submit(
        "defects",
        f"python analysis/interface_dynamics/hydrogen_bonds/02_run_row.py --manifest {manifest} --row $SLURM_ARRAY_TASK_ID --outroot results/defects",
        array=both,
    )
    submitted.append(("defects", defects))
    classifier = submit(
        "classifier",
        f"python analysis/glass_structure/opc_vitrification_spatial_workflow/07_core_awi_lda_and_free_volume.py --manifest {manifest} --outdir results/classifier",
    )
    submitted.append(("classifier", classifier))
    figures = submit(
        "figures",
        "bash workflow/reduce/figures.sh",
        dependency=f"{tg}:{energy}:{classifier}:{hbonds}:{defects}",
    )
    submitted.append(("figures", figures))
    return submitted


def main(argv=None):
    """Submit the array graph.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments without the program name.

    Returns
    -------
    int
        Zero when the jobs were submitted.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="config/manifest.tsv")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    from opcfilm.metadata import require_clean_worktree

    require_clean_worktree(Path.cwd(), allow_dirty=args.allow_dirty)
    for stage, job in submit_graph(args.manifest):
        print(f"{stage} {job}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
