"""Whole-film and regional relaxation.

The production scripts are ``analysis/glass_transition/03_fs_alpha.py`` and
``analysis/interface_dynamics/analyze_v2.py``.
"""

from opcfilm.stages import run as _run


def run(row, root):
    """Run the regional-dynamics stage for one manifest row.

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
        The stage is the relaxation script, not a package stand-in.
    """
    _run("s05", row, root)
