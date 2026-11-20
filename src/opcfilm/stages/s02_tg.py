"""Dynamic glass temperature from the hold relaxation times.

This is a reduction. It runs after the relaxation times exist. The production
script is ``analysis/glass_transition/05_fit_relaxation_tg.py``.
"""

from opcfilm.stages import run as _run


def run(row, root):
    """Run the glass-temperature stage.

    Parameters
    ----------
    row : dict
        Manifest row. The fit uses every hold temperature, not one row.
    root : path-like
        Directory that contains ``results/``.

    Returns
    -------
    None
        This function does not return.

    Raises
    ------
    NotImplementedError
        The stage is the Arrhenius script, not a package stand-in.
    """
    _run("s02", row, root)
