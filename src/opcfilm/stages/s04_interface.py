"""Capillary spectrum of a hold.

The production script is ``analysis/cooling_path/05_capillary_modes.py``.
"""

from opcfilm.stages import run as _run


def run(row, root):
    """Run the interface stage for one manifest row.

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
        The stage is the capillary script, not a package stand-in.
    """
    _run("s04", row, root)
