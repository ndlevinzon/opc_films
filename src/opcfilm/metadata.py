"""Provenance written beside every stage output."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from opcfilm.config import config_sha256


def require_clean_worktree(root, allow_dirty=False):
    """Refuse a missing or dirty git checkout.

    Parameters
    ----------
    root : path-like
        Repository directory.
    allow_dirty : bool, optional
        When true, a missing or dirty checkout is allowed. The published
        metadata then has to say so.

    Returns
    -------
    str
        Commit hash when the checkout is clean.

    Raises
    ------
    SystemExit
        If git is missing, the directory is not a checkout, or the
        worktree is dirty, and ``allow_dirty`` is false.
    """
    if allow_dirty:
        return git_commit(root)
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise SystemExit(f"git is not available ({exc}). Pass --allow-dirty to run anyway.") from exc
    if revision.returncode != 0 or not revision.stdout.strip():
        raise SystemExit("This directory is not a git checkout. Pass --allow-dirty to run anyway.")
    if status.returncode != 0 or status.stdout.strip():
        raise SystemExit("The worktree is dirty. Pass --allow-dirty to run anyway.")
    return revision.stdout.strip()


def git_commit(root):
    """Current git commit, or an empty string when git is unavailable.

    Parameters
    ----------
    root : path-like
        Repository directory.

    Returns
    -------
    str
        Commit hash, or ``""``.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def software_versions():
    """Versions of the libraries this run imported.

    Returns
    -------
    dict
        Package name to version. A missing package is omitted.
    """
    versions = {}
    for name in ("numpy", "scipy", "pandas", "MDAnalysis", "sklearn"):
        try:
            module = __import__(name)
        except ImportError:
            continue
        versions[name] = getattr(module, "__version__", "")
    return versions


def write_run_metadata(directory, row, seeds, started, root, overrides=()):
    """Write ``run_metadata.json`` in one output directory.

    Parameters
    ----------
    directory : path-like
        Stage output directory.
    row : dict
        Manifest row that produced the output.
    seeds : dict
        Seeds used by the stage.
    started : float
        ``time.time()`` at the start of the stage.
    root : path-like
        Repository directory, used for the git commit.
    overrides : sequence, optional
        ``(key, value)`` pairs that replaced sheet values.

    Returns
    -------
    pathlib.Path
        The metadata file.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "git_commit": git_commit(root),
        "config_sha256": config_sha256(),
        "manifest_row": row,
        "seeds": seeds,
        "overrides": [list(item) for item in overrides],
        "software_versions": software_versions(),
        "wall_time_s": time.time() - float(started),
    }
    path = directory / "run_metadata.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
