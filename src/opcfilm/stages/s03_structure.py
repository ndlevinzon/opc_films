"""Regional order, CHILL+, and the 300 K versus 78 K classifier.

The production script is
``analysis/glass_structure/opc_vitrification_spatial_workflow/07_core_awi_lda_and_free_volume.py``.
"""

from opcfilm.stages import run as _run


def run(row, root):
    """Run the structure stage for one manifest row.

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
        The stage is the structure script, not a package stand-in.
    """
    _run("s03", row, root)
