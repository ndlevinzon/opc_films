"""Density profiles and Gibbs surfaces.

The production scripts are ``analysis/cooling_path/01_analyze_chunk.py`` and
``04_summarize.py``.
"""

from opcfilm.stages import run as _run


def run(row, root):
    """Run the preprocess stage for one manifest row.

    Parameters
    ----------
    row : dict
        Manifest row.
    root : path-like
        Directory that contains ``results/``.

    Returns
    -------
    None
        This function does not return.

    Raises
    ------
    NotImplementedError
        The stage is the cooling-path script, not a package stand-in.
    """
    _run("s00", row, root)
