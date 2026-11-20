"""Run one of the existing manuscript scripts.

A figure module names the script. It does not draw a stand-in series when
that script's inputs are absent.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def repository():
    """Repository root.

    Returns
    -------
    pathlib.Path
        Directory that contains ``config/project.yaml``.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "config" / "project.yaml").is_file():
            return parent
    raise FileNotFoundError("config/project.yaml was not found.")


def draw(script, argv, inputs=None):
    """Run a manuscript script.

    Parameters
    ----------
    script : path-like
        Script path relative to the repository, or an absolute path.
    argv : sequence of str
        Arguments for that script, not including the program name.
    inputs : dict, optional
        Reduced tables the figure reads. A missing path is reported and
        the script is not started.

    Returns
    -------
    int
        The script's exit status.

    Raises
    ------
    SystemExit
        If no arguments were passed. The script is not run with an empty
        command that would draw nothing or the wrong figure.
    """
    missing = [
        f"{name}: {path}"
        for name, path in (inputs or {}).items()
        if not Path(path).is_file() and not (repository() / path).is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing figure inputs: " + "; ".join(missing))
    if not argv:
        raise SystemExit(f"Pass the arguments for {script}.")
    path = Path(script)
    if not path.is_file():
        path = repository() / script
    return subprocess.call([sys.executable, str(path), *list(argv)])
