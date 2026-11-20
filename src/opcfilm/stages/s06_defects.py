"""Hydrogen-bond retention, free OH, and coordination defects.

The production scripts are ``analysis/interface_dynamics/hbond_v3.py`` and
``analysis/interface_dynamics/hydrogen_bonds/03_analyze_free_oh.py``.
"""

from opcfilm.stages import run as _run


def run(row, root):
    """Run the defect stage for one manifest row.

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
        The stage is the hydrogen-bond script, not a package stand-in.
    """
    _run("s06", row, root)
