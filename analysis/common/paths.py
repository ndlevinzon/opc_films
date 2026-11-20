"""Locate the analysis package without hard-coded install paths."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_analysis_on_path(caller_file):
    """Insert the ``analysis`` directory into ``sys.path``.

    Parameters
    ----------
    caller_file : path-like
        ``__file__`` of the script that wants to import ``common``.

    Returns
    -------
    pathlib.Path
        Directory that contains the ``common`` package.
    """
    for parent in Path(caller_file).resolve().parents:
        if (parent / "common" / "__init__.py").is_file():
            entry = str(parent)
            if entry not in sys.path:
                sys.path.insert(0, entry)
            return parent
    raise RuntimeError(
        "Could not find the analysis/common package above " + str(caller_file)
    )
